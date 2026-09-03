// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Foyer-backed cache for immutable Lance data-file reads.

use std::collections::{BTreeMap, HashMap};
use std::fmt::{Debug, Display, Formatter};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use foyer::HybridCache;
use futures::StreamExt;
use futures::stream::BoxStream;
use lance_io::object_store::WrappingObjectStore;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    RenameOptions, Result,
};

use crate::data_cache::{DataCacheFactory, DatasetDataCache, LanceDataCacheStatistics};
const CACHE_KEY_VERSION: &str = "lance-data-v1";
// Bound concurrent disk-tier lookups so one large range request cannot flood
// Foyer's storage executor or starve other Lance queries.
const DATA_CACHE_LOOKUP_CONCURRENCY: usize = 8;

/// Process-local owner of a Foyer hybrid cache.
#[derive(Clone)]
pub(crate) struct FoyerDataCache {
    cache: HybridCache<String, Bytes>,
    read_block_size: usize,
    wrapped_stores: Arc<Mutex<HashMap<usize, WrappedStore>>>,
}

struct WrappedStore {
    wrapper: Weak<dyn ObjectStore>,
    origin: Weak<dyn ObjectStore>,
}

impl Debug for FoyerDataCache {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoyerDataCache")
            .field("read_block_size", &self.read_block_size)
            .finish_non_exhaustive()
    }
}

impl FoyerDataCache {
    #[cfg(test)]
    async fn try_new(
        directory: &std::path::Path,
        disk_capacity: usize,
        read_block_size: usize,
    ) -> std::result::Result<Self, foyer::Error> {
        let cache =
            crate::foyer_cache::build_disk_cache(directory, disk_capacity, 2 * read_block_size)
                .await?;
        Ok(Self::from_cache(cache, read_block_size))
    }

    pub(crate) fn from_cache(cache: HybridCache<String, Bytes>, read_block_size: usize) -> Self {
        Self {
            cache,
            read_block_size,
            wrapped_stores: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn is_cacheable_data_file(location: &Path) -> bool {
        let mut parts = location.as_ref().rsplit('/');
        matches!(
            (parts.next(), parts.next()),
            (Some(file), Some("data")) if file.ends_with(".lance")
        )
    }

    fn key(&self, store_prefix: &str, location: &Path, block_index: u64) -> String {
        format!(
            "{CACHE_KEY_VERSION}\0{}\0{store_prefix}\0{}\0{block_index}",
            self.read_block_size,
            location.as_ref()
        )
    }

    fn size_key(&self, store_prefix: &str, location: &Path) -> String {
        format!(
            "{CACHE_KEY_VERSION}\0{}\0{store_prefix}\0{}\0size",
            self.read_block_size,
            location.as_ref()
        )
    }

    fn create_scope(&self) -> Arc<DatasetFoyerDataCache> {
        Arc::new(DatasetFoyerDataCache {
            cache: self.clone(),
            statistics: Arc::new(FoyerDataCacheStatistics::default()),
        })
    }

    fn unwrap_store(&self, store: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        let identity = Arc::as_ptr(&store) as *const () as usize;
        let origin = {
            let mut wrapped_stores = self
                .wrapped_stores
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let origin = wrapped_stores.get(&identity).and_then(|entry| {
                let wrapper = entry.wrapper.upgrade()?;
                if Arc::ptr_eq(&wrapper, &store) {
                    entry.origin.upgrade()
                } else {
                    None
                }
            });
            if origin.is_none() {
                wrapped_stores.remove(&identity);
            }
            origin
        };
        match origin {
            Some(origin) => origin,
            None => store,
        }
    }

    fn remember_wrapper(&self, wrapper: &Arc<dyn ObjectStore>, origin: &Arc<dyn ObjectStore>) {
        let identity = Arc::as_ptr(wrapper) as *const () as usize;
        self.wrapped_stores
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(
                identity,
                WrappedStore {
                    wrapper: Arc::downgrade(wrapper),
                    origin: Arc::downgrade(origin),
                },
            );
    }

    fn forget_wrapper(&self, identity: usize) {
        self.wrapped_stores
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&identity);
    }
}

#[derive(Debug, Default)]
struct FoyerDataCacheStatistics {
    bytes_read_from_cache: AtomicU64,
    bytes_read_from_remote: AtomicU64,
}

impl FoyerDataCacheStatistics {
    fn record(&self, bytes_read_from_cache: u64, bytes_read_from_remote: u64) {
        self.bytes_read_from_cache
            .fetch_add(bytes_read_from_cache, Ordering::Relaxed);
        self.bytes_read_from_remote
            .fetch_add(bytes_read_from_remote, Ordering::Relaxed);
    }

    fn snapshot(&self) -> LanceDataCacheStatistics {
        LanceDataCacheStatistics {
            bytes_read_from_cache: self.bytes_read_from_cache.load(Ordering::Relaxed),
            bytes_read_from_remote: self.bytes_read_from_remote.load(Ordering::Relaxed),
        }
    }
}

impl DataCacheFactory for FoyerDataCache {
    fn attach(&self, dataset: lance::Dataset) -> (lance::Dataset, Arc<dyn DatasetDataCache>) {
        let scope = self.create_scope();
        let wrapper: Arc<dyn WrappingObjectStore> = scope.clone();
        let dataset = dataset.with_object_store_wrappers([wrapper]);
        (dataset, scope)
    }
}

#[derive(Debug)]
struct DatasetFoyerDataCache {
    cache: FoyerDataCache,
    statistics: Arc<FoyerDataCacheStatistics>,
}

impl DatasetDataCache for DatasetFoyerDataCache {
    fn snapshot(&self) -> LanceDataCacheStatistics {
        self.statistics.snapshot()
    }

    fn attach_fresh(&self, dataset: lance::Dataset) -> (lance::Dataset, Arc<dyn DatasetDataCache>) {
        self.cache.attach(dataset)
    }
}

impl WrappingObjectStore for DatasetFoyerDataCache {
    fn wrap(&self, store_prefix: &str, original: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        // A derived Dataset can already contain this cache wrapper. Resolve
        // that exact wrapper back to its origin before attaching fresh
        // dataset-scoped counters.
        let original = self.cache.unwrap_store(original);
        let reader = DataCacheReader {
            cache: self.cache.clone(),
            store_prefix: store_prefix.to_owned(),
            original: original.clone(),
            statistics: self.statistics.clone(),
        };
        let cached_store =
            Arc::new_cyclic(|weak: &Weak<DataCacheObjectStore>| DataCacheObjectStore {
                reader,
                identity: weak.as_ptr() as usize,
            });
        let wrapped: Arc<dyn ObjectStore> = cached_store.clone();
        self.cache.remember_wrapper(&wrapped, &original);
        wrapped
    }
}

#[derive(Debug)]
struct DataCacheObjectStore {
    reader: DataCacheReader,
    identity: usize,
}

#[derive(Clone, Debug)]
struct DataCacheReader {
    cache: FoyerDataCache,
    store_prefix: String,
    original: Arc<dyn ObjectStore>,
    statistics: Arc<FoyerDataCacheStatistics>,
}

impl Drop for DataCacheObjectStore {
    fn drop(&mut self) {
        self.reader.cache.forget_wrapper(self.identity);
    }
}

impl Display for DataCacheObjectStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FoyerDataCache({})", self.reader.original)
    }
}

impl DataCacheObjectStore {
    fn is_cache_safe_get(options: &GetOptions) -> bool {
        !options.head
            && options.if_match.is_none()
            && options.if_none_match.is_none()
            && options.if_modified_since.is_none()
            && options.if_unmodified_since.is_none()
            && options.version.is_none()
            && options.extensions.is_empty()
    }

    async fn cached_get(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        // Fetch metadata separately so the returned GetResult retains the origin's identity while
        // its payload uses the same block cache as get_ranges(). This also provides the object size
        // needed to resolve bounded, offset, and suffix ranges.
        let GetResult {
            meta: metadata,
            attributes,
            ..
        } = self
            .reader
            .original
            .get_opts(
                location,
                GetOptions {
                    head: true,
                    ..Default::default()
                },
            )
            .await?;
        let object_size = metadata.size;
        self.reader.cache.cache.insert(
            self.reader
                .cache
                .size_key(&self.reader.store_prefix, location),
            Bytes::copy_from_slice(&object_size.to_le_bytes()),
        );

        let range = match options.range.clone() {
            Some(requested) => match requested.as_range(object_size) {
                Ok(range) if !range.is_empty() => range,
                // Preserve the origin's exact error for invalid or empty ranges.
                _ => return self.reader.original.get_opts(location, options).await,
            },
            None => 0..object_size,
        };

        let reader = self.reader.clone();
        let stream_location = location.clone();
        let stream_range = range.clone();
        let stream = futures::stream::try_unfold(
            (reader, stream_location, stream_range),
            |(reader, location, remaining)| async move {
                if remaining.is_empty() {
                    return Ok(None);
                }

                // Yield no more than the remainder of one cache block. The next block is not
                // requested until the consumer polls again, so cancellation drops the pending
                // range without downloading or retaining the rest of the object.
                let block_size = reader.cache.read_block_size as u64;
                let bytes_to_boundary = block_size - remaining.start % block_size;
                let end = remaining
                    .start
                    .saturating_add(bytes_to_boundary)
                    .min(remaining.end);
                let chunk_range = remaining.start..end;
                let chunk = reader
                    .cached_ranges(&location, std::slice::from_ref(&chunk_range))
                    .await?
                    .into_iter()
                    .next()
                    .ok_or_else(|| cache_error(format!("missing get result for {location}")))?;
                Ok(Some((chunk, (reader, location, end..remaining.end))))
            },
        );
        let payload = GetResultPayload::Stream(Box::pin(stream));
        Ok(GetResult {
            payload,
            meta: metadata,
            range,
            attributes,
        })
    }
}

impl DataCacheReader {
    async fn read_origin_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> Result<Vec<Bytes>> {
        let bytes = self.original.get_ranges(location, ranges).await?;
        self.statistics.record(0, total_bytes(&bytes));
        Ok(bytes)
    }

    async fn object_size(&self, location: &Path) -> Result<u64> {
        let key = self.cache.size_key(&self.store_prefix, location);
        match self.cache.cache.get(&key).await {
            Ok(Some(entry)) => match entry.value().as_ref().try_into() {
                Ok(bytes) => return Ok(u64::from_le_bytes(bytes)),
                Err(_) => log::warn!(
                    "Foyer data-cache size entry was malformed for {location}; refreshing it"
                ),
            },
            Ok(None) => {}
            Err(error) => {
                log::warn!("Foyer data-cache size lookup failed for {location}: {error}");
            }
        }

        let size = self.original.head(location).await?.size;
        self.cache
            .cache
            .insert(key, Bytes::copy_from_slice(&size.to_le_bytes()));
        Ok(size)
    }

    async fn cached_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        if ranges.iter().any(|range| range.start >= range.end) {
            return self.read_origin_ranges(location, ranges).await;
        }

        let object_size = self.object_size(location).await?;
        if ranges.iter().any(|range| range.start >= object_size) {
            // Preserve the origin's exact error for ranges that start at or
            // beyond EOF.
            return self.read_origin_ranges(location, ranges).await;
        }
        let readable_ranges = ranges
            .iter()
            .map(|range| range.start..range.end.min(object_size))
            .collect::<Vec<_>>();

        let block_size = self.cache.read_block_size as u64;
        let mut blocks = BTreeMap::<u64, Option<Bytes>>::new();
        for range in &readable_ranges {
            let first = range.start / block_size;
            let last = (range.end - 1) / block_size;
            for block_index in first..=last {
                blocks.entry(block_index).or_default();
            }
        }

        // Data entries are disk-only. Look up independent blocks concurrently,
        // but keep a fixed bound to avoid flooding the storage executor.
        let cache = self.cache.cache.clone();
        let lookup_items = blocks
            .keys()
            .copied()
            .map(|block_index| {
                (
                    block_index,
                    self.cache.key(&self.store_prefix, location, block_index),
                )
            })
            .collect::<Vec<_>>();
        let lookup_concurrency = lookup_items.len().clamp(1, DATA_CACHE_LOOKUP_CONCURRENCY);
        let mut lookup_stream =
            futures::stream::iter(lookup_items.into_iter().map(|(block_index, key)| {
                let cache = cache.clone();
                async move { (block_index, cache.get(&key).await) }
            }))
            .buffer_unordered(lookup_concurrency);
        while let Some((block_index, result)) = lookup_stream.next().await {
            match result {
                Ok(Some(entry)) => {
                    let expected_size = (object_size - block_index * block_size).min(block_size);
                    if entry.value().len() as u64 != expected_size {
                        // Reject the whole malformed block, even if its short
                        // contents would cover this particular requested slice.
                        self.cache.cache.remove(&self.cache.key(
                            &self.store_prefix,
                            location,
                            block_index,
                        ));
                        continue;
                    }
                    if let Some(block) = blocks.get_mut(&block_index) {
                        *block = Some(entry.value().clone());
                    }
                }
                Ok(None) => {}
                Err(error) => {
                    // Cache availability must not affect query correctness.
                    log::warn!("Foyer data-cache lookup failed for {location}: {error}");
                }
            }
        }

        let (bytes_read_from_cache, bytes_read_from_remote) =
            requested_bytes_by_cache_status(&readable_ranges, block_size, &blocks);
        let missing: Vec<u64> = blocks
            .iter()
            .filter_map(|(block_index, value)| value.is_none().then_some(*block_index))
            .collect();
        let mut runs = Vec::<Range<u64>>::new();
        for block_index in missing {
            let start = block_index
                .checked_mul(block_size)
                .ok_or_else(|| cache_error("data-cache block offset overflow"))?;
            let end = start
                .checked_add(block_size)
                .ok_or_else(|| cache_error("data-cache block end overflow"))?
                .min(object_size);
            match runs.last_mut() {
                Some(run) if run.end == start => run.end = end,
                _ => runs.push(start..end),
            }
        }

        if !runs.is_empty() {
            // Fetch all contiguous miss runs together so a large Lance read is
            // not expanded into one remote request per cache block.
            let fetched = self.original.get_ranges(location, &runs).await?;
            for (run, bytes) in runs.into_iter().zip(fetched) {
                let first_block = run.start / block_size;
                for (offset, chunk) in bytes.chunks(self.cache.read_block_size).enumerate() {
                    let block_index = first_block + offset as u64;
                    let value = Bytes::copy_from_slice(chunk);
                    let key = self.cache.key(&self.store_prefix, location, block_index);
                    self.cache.cache.insert(key, value.clone());
                    if let Some(block) = blocks.get_mut(&block_index) {
                        *block = Some(value);
                    }
                }
            }
        }

        let assembled = readable_ranges
            .iter()
            .map(|range| assemble_range(location, range, block_size, &blocks))
            .collect::<Result<Vec<_>>>();
        match assembled {
            Ok(bytes) => {
                self.statistics
                    .record(bytes_read_from_cache, bytes_read_from_remote);
                Ok(bytes)
            }
            Err(error) => {
                // A malformed or incomplete cached entry must never turn a
                // valid source read into a query failure.
                log::warn!(
                    "Foyer data-cache entry was unusable for {location}; bypassing cache: {error}"
                );
                let bytes = self.original.get_ranges(location, ranges).await?;
                self.statistics.record(0, total_bytes(&bytes));
                Ok(bytes)
            }
        }
    }
}

fn total_bytes(ranges: &[Bytes]) -> u64 {
    ranges.iter().fold(0_u64, |total, bytes| {
        total.saturating_add(bytes.len() as u64)
    })
}

fn requested_bytes_by_cache_status(
    ranges: &[Range<u64>],
    block_size: u64,
    blocks: &BTreeMap<u64, Option<Bytes>>,
) -> (u64, u64) {
    let mut hit_bytes = 0_u64;
    let mut miss_bytes = 0_u64;
    for range in ranges {
        let mut start = range.start;
        while start < range.end {
            let block_index = start / block_size;
            let block_start = block_index * block_size;
            let end = range.end.min(block_start.saturating_add(block_size));
            let bytes = end - start;
            if blocks
                .get(&block_index)
                .is_some_and(|block| block.is_some())
            {
                hit_bytes = hit_bytes.saturating_add(bytes);
            } else {
                miss_bytes = miss_bytes.saturating_add(bytes);
            }
            start = end;
        }
    }
    (hit_bytes, miss_bytes)
}

fn assemble_range(
    location: &Path,
    range: &Range<u64>,
    block_size: u64,
    blocks: &BTreeMap<u64, Option<Bytes>>,
) -> Result<Bytes> {
    if range.is_empty() {
        return Ok(Bytes::new());
    }
    // The caller already clamps the range to the object's EOF. A short cached value
    // must not shorten it again: reject incomplete coverage so cached_ranges can
    // retry the original request against the source store.
    let first = range.start / block_size;
    let last = (range.end - 1) / block_size;
    if first == last {
        let block = blocks
            .get(&first)
            .and_then(Option::as_ref)
            .ok_or_else(|| cache_error(format!("missing block {first} for {location}")))?;
        let block_start = first * block_size;
        let start = usize::try_from(range.start - block_start)
            .map_err(|_| cache_error("data-cache slice start exceeds usize::MAX"))?;
        let end = usize::try_from((range.end - block_start).min(block_size))
            .map_err(|_| cache_error("data-cache slice end exceeds usize::MAX"))?;
        if end > block.len() {
            return Err(cache_error(format!(
                "short data-cache block {first} for {location}: need {start}..{end}, got {} bytes",
                block.len()
            )));
        }
        return Ok(block.slice(start..end));
    }

    let requested_len = usize::try_from(range.end - range.start)
        .map_err(|_| cache_error(format!("range {range:?} for {location} exceeds usize::MAX")))?;
    let mut output = BytesMut::with_capacity(requested_len);
    for block_index in first..=last {
        let block = blocks
            .get(&block_index)
            .and_then(Option::as_ref)
            .ok_or_else(|| cache_error(format!("missing block {block_index} for {location}")))?;
        let block_start = block_index * block_size;
        let start = usize::try_from(range.start.saturating_sub(block_start))
            .map_err(|_| cache_error("data-cache slice start exceeds usize::MAX"))?;
        let end_in_block = range.end.saturating_sub(block_start).min(block_size);
        let end = usize::try_from(end_in_block)
            .map_err(|_| cache_error("data-cache slice end exceeds usize::MAX"))?;
        if end > block.len() {
            return Err(cache_error(format!(
                "short data-cache block {block_index} for {location}: need {start}..{end}, got {} bytes",
                block.len()
            )));
        }
        output.extend_from_slice(&block[start..end]);
    }
    if output.len() != requested_len {
        return Err(cache_error(format!(
            "incomplete data-cache range {range:?} for {location}: expected {requested_len} bytes, got {}",
            output.len()
        )));
    }
    Ok(output.freeze())
}

fn cache_error(message: impl Into<String>) -> object_store::Error {
    object_store::Error::Generic {
        store: "foyer_data_cache",
        source: Box::new(std::io::Error::other(message.into())),
    }
}

#[async_trait]
#[deny(clippy::missing_trait_methods)]
impl ObjectStore for DataCacheObjectStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        self.reader.original.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.reader
            .original
            .put_multipart_opts(location, opts)
            .await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        if FoyerDataCache::is_cacheable_data_file(location) && Self::is_cache_safe_get(&options) {
            self.cached_get(location, options).await
        } else {
            self.reader.original.get_opts(location, options).await
        }
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        if FoyerDataCache::is_cacheable_data_file(location) {
            self.reader.cached_ranges(location, ranges).await
        } else {
            self.reader.original.get_ranges(location, ranges).await
        }
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        self.reader.original.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.reader.original.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        self.reader.original.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.reader.original.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        self.reader.original.copy_opts(from, to, options).await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, options: RenameOptions) -> Result<()> {
        self.reader.original.rename_opts(from, to, options).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use futures::StreamExt;
    use lance_io::object_store::ChainedWrappingObjectStore;
    use object_store::GetRange;
    use object_store::memory::InMemory;

    use super::*;

    fn wrap_for_test(
        cache: &FoyerDataCache,
        original: Arc<dyn ObjectStore>,
    ) -> (Arc<dyn ObjectStore>, Arc<DatasetFoyerDataCache>) {
        let scope = cache.create_scope();
        (scope.wrap("memory://test", original), scope)
    }

    #[test]
    fn assemble_range_rejects_incomplete_cached_coverage() {
        let location = Path::from("table.lance/data/part-0.lance");
        let full_block = Bytes::from_static(b"abcdefgh");
        // Exercise a single block (including a start beyond the cached bytes),
        // every position in a multi-block range, and an empty block after output.
        for (range, short_index, short_len) in [
            (0..8, 0, 4),
            (2..8, 0, 4),
            (4..8, 0, 4),
            (6..8, 0, 4),
            (0..8, 0, 0),
            (0..24, 0, 4),
            (0..24, 1, 4),
            (0..24, 2, 4),
            (0..24, 1, 0),
        ] {
            let mut blocks = BTreeMap::from([
                (0, Some(full_block.clone())),
                (1, Some(full_block.clone())),
                (2, Some(full_block.clone())),
            ]);
            blocks.insert(short_index, Some(full_block.slice(..short_len)));
            let error = assemble_range(&location, &range, 8, &blocks).unwrap_err();
            assert!(
                error.to_string().contains("short data-cache block"),
                "range={range:?}, short_index={short_index}, short_len={short_len}: {error}"
            );
        }
    }

    #[test]
    fn assemble_range_preserves_exact_slices_and_eof_tail() {
        let location = Path::from("table.lance/data/part-0.lance");
        let data = Bytes::from_static(b"abcdefghijk");
        let blocks = BTreeMap::from([(0, Some(data.slice(..8))), (1, Some(data.slice(8..)))]);
        // The last block is shorter than the cache block size but fully covers
        // each requested slice. Rejecting every short value would break EOF reads.
        for range in [0..0, 0..8, 2..6, 3..11, 8..11, 9..11] {
            assert_eq!(
                assemble_range(&location, &range, 8, &blocks).unwrap(),
                data.slice(range.start as usize..range.end as usize),
                "range={range:?}"
            );
        }
        assert!(
            assemble_range(&location, &(0..0), 8, &BTreeMap::new())
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn incomplete_cache_hits_fall_back_to_complete_source_ranges() {
        const BLOCK_SIZE: usize = 64 * 1024;
        let original = Arc::new(InMemory::new());
        let data = Bytes::from(
            (0..3 * BLOCK_SIZE)
                .map(|value| value as u8)
                .collect::<Vec<_>>(),
        );

        for (case, range, short_index, short_len) in [
            ("single", 0..BLOCK_SIZE as u64, 0, BLOCK_SIZE / 2),
            ("first", 0..data.len() as u64, 0, BLOCK_SIZE / 2),
            ("middle", 0..data.len() as u64, 1, BLOCK_SIZE / 2),
            ("last", 0..data.len() as u64, 2, BLOCK_SIZE / 2),
            ("empty", 0..data.len() as u64, 1, 0),
            ("partial", 0..16, 0, BLOCK_SIZE / 2),
        ] {
            let directory = tempfile::tempdir().unwrap();
            // Keep every injected entry available to exercise malformed hits,
            // rather than ordinary misses caused by eviction.
            let cache = FoyerDataCache::try_new(directory.path(), 64 * BLOCK_SIZE, BLOCK_SIZE)
                .await
                .unwrap();
            let location = Path::from(format!("table.lance/data/{case}.lance"));
            original.put(&location, data.clone().into()).await.unwrap();
            cache.cache.insert(
                cache.size_key("memory://test", &location),
                Bytes::copy_from_slice(&(data.len() as u64).to_le_bytes()),
            );
            for block_index in 0..3 {
                let value = if block_index == short_index {
                    Bytes::from(vec![255; short_len])
                } else {
                    let start = block_index * BLOCK_SIZE;
                    data.slice(start..start + BLOCK_SIZE)
                };
                cache.cache.insert(
                    cache.key("memory://test", &location, block_index as u64),
                    value,
                );
            }
            // Recover fully persisted corruption. This also avoids racing the
            // fixture's pending writes against the cache's repair writes.
            cache.cache.close().await.unwrap();
            drop(cache);
            crate::foyer_cache::wait_for_directory_release(directory.path()).await;
            let cache = FoyerDataCache::try_new(directory.path(), 64 * BLOCK_SIZE, BLOCK_SIZE)
                .await
                .unwrap();
            let short_key = cache.key("memory://test", &location, short_index as u64);
            assert_eq!(
                cache
                    .cache
                    .get(&short_key)
                    .await
                    .unwrap()
                    .unwrap()
                    .value()
                    .len(),
                short_len
            );
            let (wrapped, statistics) = wrap_for_test(&cache, original.clone());
            assert_eq!(
                wrapped
                    .get_ranges(&location, std::slice::from_ref(&range))
                    .await
                    .unwrap(),
                vec![data.slice(range.start as usize..range.end as usize)],
                "case={case}"
            );
            let bad_start = (short_index * BLOCK_SIZE) as u64;
            let bad_end = bad_start + BLOCK_SIZE as u64;
            let remote_bytes = range.end.min(bad_end) - range.start.max(bad_start);
            assert_eq!(
                statistics.snapshot().bytes_read_from_cache,
                range.end - range.start - remote_bytes,
                "case={case}"
            );
            assert_eq!(
                statistics.snapshot().bytes_read_from_remote,
                remote_bytes,
                "case={case}"
            );
            // The malformed block is repaired, so the next read is fully cached.
            let before = statistics.snapshot();
            wrapped
                .get_ranges(&location, std::slice::from_ref(&range))
                .await
                .unwrap();
            assert_eq!(
                statistics.snapshot().bytes_read_from_remote,
                before.bytes_read_from_remote
            );
        }
    }

    #[tokio::test]
    async fn caches_only_immutable_data_file_ranges() {
        let directory = tempfile::tempdir().unwrap();
        let cache = FoyerDataCache::try_new(directory.path(), 1024 * 1024, 64 * 1024)
            .await
            .unwrap();
        let original = Arc::new(InMemory::new());
        let data_path = Path::from("table.lance/data/part-0.lance");
        let second_data_path = Path::from("table.lance/data/part-1.lance");
        let manifest_path = Path::from("table.lance/_versions/1.manifest");
        let data = Bytes::from((0..200_000).map(|value| value as u8).collect::<Vec<_>>());
        let second_data = Bytes::from_static(b"second fragment");
        original.put(&data_path, data.clone().into()).await.unwrap();
        original
            .put(&second_data_path, second_data.clone().into())
            .await
            .unwrap();
        original
            .put(&manifest_path, Bytes::from_static(b"manifest").into())
            .await
            .unwrap();

        let (wrapped, statistics) = wrap_for_test(&cache, original.clone());
        let ranges = vec![10..90_000, 65_000..140_000, 190_000..220_000];
        let second_data_range = 0..15;
        let first = wrapped.get_ranges(&data_path, &ranges).await.unwrap();
        assert_eq!(first[0], data.slice(10..90_000));
        assert_eq!(first[1], data.slice(65_000..140_000));
        assert_eq!(first[2], data.slice(190_000..200_000));
        assert_eq!(
            wrapped
                .get_ranges(&second_data_path, std::slice::from_ref(&second_data_range))
                .await
                .unwrap(),
            vec![second_data.clone()]
        );

        assert_eq!(
            statistics.snapshot(),
            LanceDataCacheStatistics {
                bytes_read_from_cache: 0,
                bytes_read_from_remote: 175_005,
            }
        );

        original.delete(&data_path).await.unwrap();
        original.delete(&second_data_path).await.unwrap();
        let second = wrapped.get_ranges(&data_path, &ranges).await.unwrap();
        assert_eq!(second, first);
        assert_eq!(
            wrapped
                .get_ranges(&second_data_path, std::slice::from_ref(&second_data_range))
                .await
                .unwrap(),
            vec![second_data]
        );

        assert_eq!(
            statistics.snapshot(),
            LanceDataCacheStatistics {
                bytes_read_from_cache: 175_005,
                bytes_read_from_remote: 175_005,
            }
        );

        assert_eq!(
            wrapped.get_range(&manifest_path, 0..8).await.unwrap(),
            Bytes::from_static(b"manifest")
        );
        original.delete(&manifest_path).await.unwrap();
        assert!(wrapped.get_range(&manifest_path, 0..8).await.is_err());
    }

    #[tokio::test]
    async fn caches_small_data_file_whole_object_reads() {
        let directory = tempfile::tempdir().unwrap();
        let cache = FoyerDataCache::try_new(directory.path(), 1024 * 1024, 64 * 1024)
            .await
            .unwrap();
        let original = Arc::new(InMemory::new());
        let data_path = Path::from("table.lance/data/small.lance");
        let data = Bytes::from(vec![7; 42_000]);
        original.put(&data_path, data.clone().into()).await.unwrap();

        let (wrapped, statistics) = wrap_for_test(&cache, original);
        let first = wrapped
            .get(&data_path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(first, data);
        assert_eq!(
            statistics.snapshot(),
            LanceDataCacheStatistics {
                bytes_read_from_cache: 0,
                bytes_read_from_remote: 42_000,
            }
        );

        let second = wrapped
            .get(&data_path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(second, data);
        assert_eq!(
            statistics.snapshot(),
            LanceDataCacheStatistics {
                bytes_read_from_cache: 42_000,
                bytes_read_from_remote: 42_000,
            }
        );
    }

    #[tokio::test]
    async fn streams_large_data_file_gets_with_bounded_read_ahead() {
        let directory = tempfile::tempdir().unwrap();
        let block_size = 64 * 1024;
        let cache = FoyerDataCache::try_new(directory.path(), 1024 * 1024, block_size)
            .await
            .unwrap();
        let original = Arc::new(InMemory::new());
        let data_path = Path::from("table.lance/data/large.lance");
        let data = Bytes::from(vec![7; 4 * block_size]);
        original.put(&data_path, data.clone().into()).await.unwrap();

        let (wrapped, statistics) = wrap_for_test(&cache, original);
        let result = wrapped.get(&data_path).await.unwrap();
        assert_eq!(statistics.snapshot(), LanceDataCacheStatistics::default());

        let mut stream = result.into_stream();
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first, data.slice(..block_size));
        assert_eq!(
            statistics.snapshot(),
            LanceDataCacheStatistics {
                bytes_read_from_cache: 0,
                bytes_read_from_remote: block_size as u64,
            }
        );

        drop(stream);
        assert_eq!(
            statistics.snapshot().bytes_read_from_remote,
            block_size as u64
        );
    }

    #[tokio::test]
    async fn caches_single_data_file_range_reads() {
        let directory = tempfile::tempdir().unwrap();
        let cache = FoyerDataCache::try_new(directory.path(), 1024 * 1024, 64 * 1024)
            .await
            .unwrap();
        let original = Arc::new(InMemory::new());
        let data = Bytes::from((0..100_000).map(|value| value as u8).collect::<Vec<_>>());
        let cases = [
            (
                Path::from("table.lance/data/bounded.lance"),
                GetRange::Bounded(1_000..2_000),
                1_000..2_000,
            ),
            (
                Path::from("table.lance/data/offset.lance"),
                GetRange::Offset(90_000),
                90_000..100_000,
            ),
            (
                Path::from("table.lance/data/suffix.lance"),
                GetRange::Suffix(500),
                99_500..100_000,
            ),
        ];
        for (path, _, _) in &cases {
            original.put(path, data.clone().into()).await.unwrap();
        }

        let (wrapped, statistics) = wrap_for_test(&cache, original);
        for (path, requested, expected_range) in cases {
            let expected = data.slice(expected_range.start as usize..expected_range.end as usize);
            let before = statistics.snapshot();
            let first = wrapped
                .get_opts(&path, GetOptions::new().with_range(Some(requested.clone())))
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            assert_eq!(first, expected);
            assert_eq!(
                statistics.snapshot().bytes_read_from_remote,
                before.bytes_read_from_remote + expected.len() as u64
            );

            let second = wrapped
                .get_opts(&path, GetOptions::new().with_range(Some(requested)))
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            assert_eq!(second, expected);
            assert_eq!(
                statistics.snapshot().bytes_read_from_cache,
                before.bytes_read_from_cache + expected.len() as u64
            );
        }

        let before = statistics.snapshot();
        let conditional = GetOptions::new().with_if_match(Some("wrong-etag"));
        assert!(
            wrapped
                .get_opts(&Path::from("table.lance/data/bounded.lance"), conditional)
                .await
                .is_err()
        );
        assert_eq!(statistics.snapshot(), before);

        assert!(
            wrapped
                .get_range(
                    &Path::from("table.lance/data/bounded.lance"),
                    100_000..100_001
                )
                .await
                .is_err()
        );
        assert_eq!(statistics.snapshot(), before);
    }

    #[tokio::test]
    async fn recovers_cached_data_from_disk() {
        let directory = tempfile::tempdir().unwrap();
        let original = Arc::new(InMemory::new());
        let data_path = Path::from("table.lance/data/part-0.lance");
        let data = Bytes::from((0..100_000).map(|value| value as u8).collect::<Vec<_>>());
        original.put(&data_path, data.clone().into()).await.unwrap();

        let cache = FoyerDataCache::try_new(directory.path(), 1024 * 1024, 64 * 1024)
            .await
            .unwrap();
        let (wrapped, _) = wrap_for_test(&cache, original.clone());
        let requested_range = 10..90_000;
        assert_eq!(
            wrapped
                .get_ranges(&data_path, std::slice::from_ref(&requested_range))
                .await
                .unwrap(),
            vec![data.slice(10..90_000)]
        );
        drop(wrapped);
        cache.cache.close().await.unwrap();
        drop(cache);
        crate::foyer_cache::wait_for_directory_release(directory.path()).await;

        original.delete(&data_path).await.unwrap();
        let recovered = FoyerDataCache::try_new(directory.path(), 1024 * 1024, 64 * 1024)
            .await
            .unwrap();
        let (wrapped, statistics) = wrap_for_test(&recovered, original);
        assert_eq!(
            wrapped
                .get_ranges(&data_path, std::slice::from_ref(&requested_range))
                .await
                .unwrap(),
            vec![data.slice(10..90_000)]
        );
        assert_eq!(statistics.snapshot().bytes_read_from_cache, 89_990);
        assert_eq!(statistics.snapshot().bytes_read_from_remote, 0);
    }

    #[tokio::test]
    async fn dataset_scopes_share_cache_without_sharing_statistics() {
        let directory = tempfile::tempdir().unwrap();
        let cache = FoyerDataCache::try_new(directory.path(), 1024 * 1024, 64 * 1024)
            .await
            .unwrap();
        let original = Arc::new(InMemory::new());
        let data_path = Path::from("table.lance/data/part-0.lance");
        let data = Bytes::from((0..100_000).map(|value| value as u8).collect::<Vec<_>>());
        original.put(&data_path, data.clone().into()).await.unwrap();

        let source_scope = cache.create_scope();
        let source_store = source_scope.wrap("memory://test", original.clone());

        // A restored Dataset is derived from an already-wrapped source
        // Dataset. The fresh scope must unwrap to the registered origin rather
        // than nesting over the source scope.
        let restored_scope = cache.create_scope();
        let restored_store = restored_scope.wrap("memory://test", source_store.clone());
        let requested_range = 10..90_000;
        assert_eq!(
            restored_store
                .get_ranges(&data_path, std::slice::from_ref(&requested_range))
                .await
                .unwrap(),
            vec![data.slice(10..90_000)]
        );
        assert_eq!(source_scope.snapshot(), LanceDataCacheStatistics::default());
        assert_eq!(restored_scope.snapshot().bytes_read_from_remote, 89_990);

        original.delete(&data_path).await.unwrap();
        assert_eq!(
            source_store
                .get_ranges(&data_path, std::slice::from_ref(&requested_range))
                .await
                .unwrap(),
            vec![data.slice(10..90_000)]
        );
        assert_eq!(source_scope.snapshot().bytes_read_from_cache, 89_990);
        assert_eq!(restored_scope.snapshot().bytes_read_from_remote, 89_990);

        drop(restored_store);
        drop(source_store);
        assert!(cache.wrapped_stores.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn chained_scopes_drop_intermediate_store_without_deadlocking() {
        let directory = tempfile::tempdir().unwrap();
        let cache = FoyerDataCache::try_new(directory.path(), 1024 * 1024, 64 * 1024)
            .await
            .unwrap();
        let first: Arc<dyn WrappingObjectStore> = cache.create_scope();
        let second: Arc<dyn WrappingObjectStore> = cache.create_scope();
        let chained = ChainedWrappingObjectStore::new(vec![first, second]);
        let original: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let (sender, receiver) = mpsc::channel();

        let thread = std::thread::spawn(move || {
            sender
                .send(chained.wrap("memory://test", original))
                .unwrap();
        });
        let wrapped = receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("chained cache wrappers deadlocked while dropping the intermediate store");
        thread.join().unwrap();

        assert_eq!(cache.wrapped_stores.lock().unwrap().len(), 1);
        drop(wrapped);
        assert!(cache.wrapped_stores.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn truncates_ranges_at_eof_before_enumerating_cache_blocks() {
        let directory = tempfile::tempdir().unwrap();
        let cache = FoyerDataCache::try_new(directory.path(), 1024 * 1024, 64 * 1024)
            .await
            .unwrap();
        let original = Arc::new(InMemory::new());
        let data_path = Path::from("table.lance/data/part-0.lance");
        let data = Bytes::from((0..100_000).map(|value| value as u8).collect::<Vec<_>>());
        original.put(&data_path, data.into()).await.unwrap();

        let requested_range = 99_990..300_000;
        let expected = original
            .get_ranges(&data_path, std::slice::from_ref(&requested_range))
            .await
            .unwrap();
        let (wrapped, statistics) = wrap_for_test(&cache, original.clone());
        let actual = wrapped
            .get_ranges(&data_path, std::slice::from_ref(&requested_range))
            .await
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(statistics.snapshot().bytes_read_from_remote, 10);

        original.delete(&data_path).await.unwrap();
        assert_eq!(
            wrapped
                .get_ranges(&data_path, std::slice::from_ref(&requested_range))
                .await
                .unwrap(),
            expected
        );
        assert_eq!(statistics.snapshot().bytes_read_from_cache, 10);
        assert_eq!(statistics.snapshot().bytes_read_from_remote, 10);
    }

    #[test]
    fn recognizes_only_direct_data_children() {
        assert!(FoyerDataCache::is_cacheable_data_file(&Path::from(
            "dataset/data/part.lance"
        )));
        assert!(!FoyerDataCache::is_cacheable_data_file(&Path::from(
            "dataset/data/nested/part.lance"
        )));
        assert!(!FoyerDataCache::is_cacheable_data_file(&Path::from(
            "dataset/indices/index.lance"
        )));
        assert!(!FoyerDataCache::is_cacheable_data_file(&Path::from(
            "dataset/_versions/1.manifest"
        )));
    }
}
