// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Two-tier cache backend for serializable Lance index entries.
//!
//! L1 is the Session's index memory cache, backed by `QuickCacheBackend`.
//! Metadata uses the Session's separate L1 cache and does not enter this backend.
//! L2 is the shared Foyer disk cache. It stores the Lance `CacheCodec` envelope,
//! so an entry written
//! by an older process is either decoded safely or treated as a miss.

use std::fmt::{Debug, Formatter};
use std::fs;
use std::future::Future;
use std::io::Write;
use std::path::{Path as FsPath, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use foyer::HybridCache;
use lance_core::Result;
use lance_core::cache::{
    CACHE_KEY_FORMAT, CacheBackend, CacheCodec, CacheDecode, CacheEntry, InternalCacheKey,
    QuickCacheBackend,
};

use crate::error::ffi_try;
use crate::session::LanceSession;

const CACHE_KEY_VERSION: &str = "lance-index-v1";
const LOGICAL_SIZE_BYTES: usize = std::mem::size_of::<u64>();
const INDEX_GENERATION_FILE: &str = "lance-index-generation";

/// Cumulative L1 index and L2 serialized-index counters for one shared session.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LanceIndexDiskCacheStats {
    pub memory_hits: u64,
    pub memory_misses: u64,
    pub disk_hits: u64,
    pub disk_misses: u64,
    pub disk_read_bytes: u64,
    pub disk_write_bytes: u64,
    pub decode_failures: u64,
    pub disk_read_errors: u64,
    pub disk_write_errors: u64,
}

#[derive(Debug, Default)]
pub(crate) struct IndexDiskCacheStats {
    memory_hits: AtomicU64,
    memory_misses: AtomicU64,
    disk_hits: AtomicU64,
    disk_misses: AtomicU64,
    disk_read_bytes: AtomicU64,
    disk_write_bytes: AtomicU64,
    decode_failures: AtomicU64,
    disk_read_errors: AtomicU64,
    disk_write_errors: AtomicU64,
}

impl IndexDiskCacheStats {
    fn snapshot(&self) -> LanceIndexDiskCacheStats {
        LanceIndexDiskCacheStats {
            memory_hits: self.memory_hits.load(Ordering::Relaxed),
            memory_misses: self.memory_misses.load(Ordering::Relaxed),
            disk_hits: self.disk_hits.load(Ordering::Relaxed),
            disk_misses: self.disk_misses.load(Ordering::Relaxed),
            disk_read_bytes: self.disk_read_bytes.load(Ordering::Relaxed),
            disk_write_bytes: self.disk_write_bytes.load(Ordering::Relaxed),
            decode_failures: self.decode_failures.load(Ordering::Relaxed),
            disk_read_errors: self.disk_read_errors.load(Ordering::Relaxed),
            disk_write_errors: self.disk_write_errors.load(Ordering::Relaxed),
        }
    }
}

/// Lance index cache with decoded values in L1 and serialized values in L2.
pub(crate) struct FoyerIndexCache {
    state: RwLock<Arc<IndexGeneration>>,
    memory_capacity: usize,
    disk: HybridCache<String, Bytes>,
    generation_path: PathBuf,
    stats: Arc<IndexDiskCacheStats>,
}

#[derive(Debug)]
struct IndexGeneration {
    memory: QuickCacheBackend,
    generation: u64,
}

impl Debug for FoyerIndexCache {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoyerIndexCache")
            .field("state", &self.state)
            .field("disk", &self.disk)
            .finish_non_exhaustive()
    }
}

impl FoyerIndexCache {
    #[cfg(test)]
    async fn try_new(
        directory: &FsPath,
        memory_capacity: usize,
        disk_capacity: usize,
        storage_block_size: usize,
    ) -> std::result::Result<Self, foyer::Error> {
        let disk =
            crate::foyer_cache::build_disk_cache(directory, disk_capacity, storage_block_size)
                .await?;
        Self::from_cache(disk, memory_capacity, directory)
    }

    pub(crate) fn from_cache(
        disk: HybridCache<String, Bytes>,
        memory_capacity: usize,
        directory: &FsPath,
    ) -> std::result::Result<Self, foyer::Error> {
        let generation = load_generation(directory)?;
        Ok(Self {
            state: RwLock::new(Arc::new(IndexGeneration {
                memory: QuickCacheBackend::with_capacity(memory_capacity),
                generation,
            })),
            memory_capacity,
            disk,
            generation_path: generation_path(directory),
            stats: Arc::new(IndexDiskCacheStats::default()),
        })
    }

    pub(crate) fn stats(&self) -> Arc<IndexDiskCacheStats> {
        self.stats.clone()
    }

    #[cfg(test)]
    fn disk_key(key: &InternalCacheKey, codec: CacheCodec) -> String {
        Self::disk_key_for_generation(key, codec, 0)
    }

    fn disk_key_for_generation(
        key: &InternalCacheKey,
        codec: CacheCodec,
        generation: u64,
    ) -> String {
        let mut encoded = String::with_capacity(32);
        for byte in key.as_bytes() {
            use std::fmt::Write;
            let _ = write!(encoded, "{byte:02x}");
        }
        format!(
            "{CACHE_KEY_VERSION}:{CACHE_KEY_FORMAT}:{generation}:{}:{encoded}",
            codec.type_id(),
        )
    }

    fn state(&self) -> Arc<IndexGeneration> {
        self.state
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    fn encode(codec: CacheCodec, entry: &CacheEntry, logical_size: usize) -> Result<Bytes> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(logical_size as u64).to_le_bytes());
        // Vec's Write implementation appends after the size prefix. A fresh
        // Cursor starts at zero and would overwrite that prefix with the codec.
        codec.serialize(entry, &mut bytes)?;
        Ok(Bytes::from(bytes))
    }

    fn decode(codec: CacheCodec, bytes: &Bytes) -> Option<(CacheEntry, usize)> {
        if bytes.len() < LOGICAL_SIZE_BYTES {
            return None;
        }
        let size = u64::from_le_bytes(bytes[..LOGICAL_SIZE_BYTES].try_into().ok()?);
        let size = usize::try_from(size).ok()?;
        let payload = bytes.slice(LOGICAL_SIZE_BYTES..);
        match codec.deserialize(&payload) {
            CacheDecode::Hit(entry) => Some((entry, size)),
            CacheDecode::Miss(_) => None,
        }
    }

    async fn load_disk(
        disk: &HybridCache<String, Bytes>,
        stats: &IndexDiskCacheStats,
        key: &str,
        codec: CacheCodec,
    ) -> Option<(CacheEntry, usize)> {
        let entry = match disk.get(key).await {
            Ok(Some(entry)) => entry,
            Ok(None) => {
                stats.disk_misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            Err(error) => {
                stats.disk_misses.fetch_add(1, Ordering::Relaxed);
                stats.disk_read_errors.fetch_add(1, Ordering::Relaxed);
                log::warn!("Foyer index cache lookup failed for {key}: {error}");
                return None;
            }
        };
        let bytes = entry.value();
        stats
            .disk_read_bytes
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        match Self::decode(codec, bytes) {
            Some(decoded) => {
                stats.disk_hits.fetch_add(1, Ordering::Relaxed);
                Some(decoded)
            }
            None => {
                stats.disk_misses.fetch_add(1, Ordering::Relaxed);
                stats.decode_failures.fetch_add(1, Ordering::Relaxed);
                disk.remove(key);
                None
            }
        }
    }

    fn store_disk(
        disk: &HybridCache<String, Bytes>,
        stats: &IndexDiskCacheStats,
        key: &str,
        codec: CacheCodec,
        entry: &CacheEntry,
        logical_size: usize,
    ) {
        match Self::encode(codec, entry, logical_size) {
            Ok(bytes) => {
                stats
                    .disk_write_bytes
                    .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                disk.insert(key.to_owned(), bytes);
            }
            Err(error) => {
                stats.disk_write_errors.fetch_add(1, Ordering::Relaxed);
                log::warn!("failed to serialize Lance index cache entry {key}: {error}");
            }
        }
    }
}

#[async_trait]
impl CacheBackend for FoyerIndexCache {
    async fn get(&self, key: &InternalCacheKey, codec: Option<CacheCodec>) -> Option<CacheEntry> {
        let state = self.state();
        if let Some(entry) = state.memory.get(key, None).await {
            self.stats.memory_hits.fetch_add(1, Ordering::Relaxed);
            return Some(entry);
        }
        self.stats.memory_misses.fetch_add(1, Ordering::Relaxed);
        let codec = codec?;
        let disk_key = Self::disk_key_for_generation(key, codec, state.generation);
        let (entry, size) = Self::load_disk(&self.disk, &self.stats, &disk_key, codec).await?;
        state.memory.insert(key, entry.clone(), size, None).await;
        Some(entry)
    }

    async fn insert(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        codec: Option<CacheCodec>,
    ) {
        let state = self.state();
        state
            .memory
            .insert(key, entry.clone(), size_bytes, None)
            .await;
        if let Some(codec) = codec {
            let disk_key = Self::disk_key_for_generation(key, codec, state.generation);
            Self::store_disk(
                &self.disk,
                &self.stats,
                &disk_key,
                codec,
                &entry,
                size_bytes,
            );
        }
    }

    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
        codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)> {
        let key = *key;
        let disk = self.disk.clone();
        let stats = self.stats.clone();
        let disk_codec = codec;
        let state = self.state();
        let generation = state.generation;
        let disk_hit = AtomicBool::new(false);
        let disk_hit_ref = &disk_hit;
        let loader = async move {
            if let Some(codec) = disk_codec {
                let disk_key = Self::disk_key_for_generation(&key, codec, generation);
                if let Some((entry, size)) = Self::load_disk(&disk, &stats, &disk_key, codec).await
                {
                    disk_hit_ref.store(true, Ordering::Relaxed);
                    return Ok((entry, size));
                }
                let result = loader.await?;
                Self::store_disk(&disk, &stats, &disk_key, codec, &result.0, result.1);
                Ok(result)
            } else {
                loader.await
            }
        };
        let result = state
            .memory
            .get_or_insert(&key, Box::pin(loader), None)
            .await;
        match &result {
            Ok((_, true)) => {
                self.stats.memory_hits.fetch_add(1, Ordering::Relaxed);
            }
            Ok((_, false)) | Err(_) => {
                self.stats.memory_misses.fetch_add(1, Ordering::Relaxed);
            }
        }
        // The inner result describes L1 only; the CacheBackend contract
        // describes whether the original loader was skipped across both tiers.
        result.map(|(entry, memory_hit)| (entry, memory_hit || disk_hit.load(Ordering::Relaxed)))
    }

    async fn clear(&self) {
        let mut state = self
            .state
            .write()
            .unwrap_or_else(|error| error.into_inner());
        let next_generation = new_generation();
        if let Err(error) = persist_generation_file(&self.generation_path, next_generation) {
            log::warn!(
                "failed to persist Foyer index cache generation; using a fresh in-process namespace: {error}"
            );
        }
        // Replace L1 with the namespace. In-flight loads retain the old state
        // and cannot repopulate the new memory cache or its disk namespace.
        *state = Arc::new(IndexGeneration {
            memory: QuickCacheBackend::with_capacity(self.memory_capacity),
            generation: next_generation,
        });
    }

    async fn num_entries(&self) -> usize {
        self.state().memory.num_entries().await
    }

    async fn size_bytes(&self) -> usize {
        self.state().memory.size_bytes().await
    }

    fn approx_num_entries(&self) -> usize {
        self.state().memory.approx_num_entries()
    }

    fn approx_size_bytes(&self) -> usize {
        self.state().memory.approx_size_bytes()
    }

    fn deep_size_of_entries(
        &self,
        context: &mut lance_core::deepsize::Context,
        size_of_entry: &dyn Fn(&CacheEntry, &mut lance_core::deepsize::Context) -> Option<usize>,
    ) -> Option<usize> {
        self.state()
            .memory
            .deep_size_of_entries(context, size_of_entry)
    }
}

fn generation_path(directory: &FsPath) -> PathBuf {
    directory.join(INDEX_GENERATION_FILE)
}

fn new_generation() -> u64 {
    uuid::Uuid::new_v4().as_u64_pair().0
}

fn load_generation(directory: &FsPath) -> std::result::Result<u64, foyer::Error> {
    let path = generation_path(directory);
    match fs::read_to_string(&path) {
        Ok(value) => match value.trim().parse::<u64>() {
            Ok(generation) => Ok(generation),
            Err(error) => {
                log::warn!("invalid index cache generation in {path:?}; starting cold: {error}");
                let generation = new_generation();
                persist_generation_file(&path, generation)?;
                Ok(generation)
            }
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let generation = new_generation();
            persist_generation_file(&path, generation)?;
            Ok(generation)
        }
        Err(error) => Err(foyer::Error::io_error(error)),
    }
}

fn persist_generation_file(
    path: &FsPath,
    generation: u64,
) -> std::result::Result<(), foyer::Error> {
    let temporary = path.with_file_name(format!(
        "{INDEX_GENERATION_FILE}.{}.tmp",
        uuid::Uuid::new_v4()
    ));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(generation.to_string().as_bytes())?;
        file.sync_all()?;
        fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map_err(foyer::Error::io_error)
}

/// Copy the disk-tier counters into `out_stats`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_session_get_index_disk_cache_stats(
    session: *const LanceSession,
    out_stats: *mut LanceIndexDiskCacheStats,
) -> i32 {
    ffi_try!(
        unsafe { index_disk_cache_stats_inner(session, out_stats) },
        neg
    )
}

unsafe fn index_disk_cache_stats_inner(
    session: *const LanceSession,
    out_stats: *mut LanceIndexDiskCacheStats,
) -> Result<i32> {
    if session.is_null() || out_stats.is_null() {
        return Err(lance_core::Error::invalid_input_source(
            "session and out_stats must not be NULL".into(),
        ));
    }
    let session = unsafe { &*session };
    let stats = session
        .index_disk_cache_stats
        .as_ref()
        .map(|stats| stats.snapshot())
        .unwrap_or_default();
    unsafe {
        std::ptr::write_unaligned(out_stats, stats);
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use tokio::sync::oneshot;

    use super::*;

    fn test_codec() -> CacheCodec {
        CacheCodec::new(
            "lance.test.U64",
            1,
            |value, writer| writer.write_raw(&value.downcast_ref::<u64>().unwrap().to_le_bytes()),
            |reader| {
                let raw = reader.read_raw()?;
                let bytes = raw.as_ref().try_into().map_err(|_| {
                    lance_core::Error::io(format!("expected 8 test entry bytes, got {}", raw.len()))
                })?;
                Ok(Arc::new(u64::from_le_bytes(bytes)))
            },
        )
    }

    async fn test_cache(directory: &FsPath) -> FoyerIndexCache {
        FoyerIndexCache::try_new(directory, 1024 * 1024, 1024 * 1024, 65536)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn disk_hits_preserve_layer_statistics_and_report_cached() {
        let directory = tempfile::tempdir().unwrap();
        let cache = test_cache(directory.path()).await;
        let key = InternalCacheKey::from_bytes([1; 16]);
        let codec = Some(test_codec());
        cache.insert(&key, Arc::new(42_u64), 8, codec).await;
        cache.state().memory.clear().await;

        for _ in 0..2 {
            let (value, was_cached) = cache
                .get_or_insert(
                    &key,
                    Box::pin(async { panic!("cache hit must not execute source loader") }),
                    codec,
                )
                .await
                .unwrap();
            assert_eq!(*value.downcast_ref::<u64>().unwrap(), 42);
            assert!(was_cached);
        }
        let stats = cache.stats.snapshot();
        assert_eq!((stats.memory_hits, stats.memory_misses), (1, 1));
        assert_eq!((stats.disk_hits, stats.disk_misses), (1, 0));

        let missing = InternalCacheKey::from_bytes([2; 16]);
        let (_, was_cached) = cache
            .get_or_insert(
                &missing,
                Box::pin(async { Ok((Arc::new(7_u64) as CacheEntry, 8)) }),
                codec,
            )
            .await
            .unwrap();
        assert!(!was_cached);
        let stats = cache.stats.snapshot();
        assert_eq!((stats.memory_hits, stats.memory_misses), (1, 2));
        assert_eq!((stats.disk_hits, stats.disk_misses), (1, 1));
    }

    #[tokio::test]
    async fn persisted_l2_hits_promote_to_l1_before_any_further_disk_lookup() {
        let directory = tempfile::tempdir().unwrap();
        let cache = test_cache(directory.path()).await;
        let keys = [
            InternalCacheKey::from_bytes([21; 16]),
            InternalCacheKey::from_bytes([22; 16]),
        ];
        for key in &keys {
            cache
                .insert(key, Arc::new(42_u64), 8, Some(test_codec()))
                .await;
        }
        cache.disk.close().await.unwrap();
        drop(cache);
        crate::foyer_cache::wait_for_directory_release(directory.path()).await;

        // Both capacities are nonzero, but neither L1 nor Foyer's pending-write
        // buffers survive reopening. A hit must read a persisted disk entry.
        let cache = test_cache(directory.path()).await;
        assert_eq!(cache.num_entries().await, 0);
        for (index, key) in keys.iter().enumerate() {
            let reads = cache.disk.storage().statistics().disk_read_ios();
            let value = if index == 0 {
                cache.get(key, Some(test_codec())).await.unwrap()
            } else {
                let (value, cached) = cache
                    .get_or_insert(
                        key,
                        Box::pin(async { panic!("persisted L2 hit must skip the source") }),
                        Some(test_codec()),
                    )
                    .await
                    .unwrap();
                assert!(cached);
                value
            };
            assert_eq!(*value.downcast_ref::<u64>().unwrap(), 42);
            assert!(cache.disk.storage().statistics().disk_read_ios() > reads);
            assert!(cache.state().memory.get(key, None).await.is_some());

            // Remove the lower-tier copy: both read APIs must now use L1,
            // without inspecting L2 or polling their source loader.
            let physical = FoyerIndexCache::disk_key_for_generation(
                key,
                test_codec(),
                cache.state().generation,
            );
            cache.disk.remove(&physical);
            cache.disk.storage().wait().await;
            let before = cache.stats.snapshot();
            let reads = cache.disk.storage().statistics().disk_read_ios();
            assert_eq!(
                *cache
                    .get(key, Some(test_codec()))
                    .await
                    .unwrap()
                    .downcast_ref::<u64>()
                    .unwrap(),
                42
            );
            let (value, cached) = cache
                .get_or_insert(
                    key,
                    Box::pin(async { panic!("L1 hit must skip the source") }),
                    Some(test_codec()),
                )
                .await
                .unwrap();
            assert!(cached);
            assert_eq!(*value.downcast_ref::<u64>().unwrap(), 42);
            let after = cache.stats.snapshot();
            assert_eq!(after.memory_hits, before.memory_hits + 2);
            assert_eq!(after.memory_misses, before.memory_misses);
            assert_eq!(after.disk_hits, before.disk_hits);
            assert_eq!(after.disk_misses, before.disk_misses);
            assert_eq!(cache.disk.storage().statistics().disk_read_ios(), reads);
        }
    }

    #[tokio::test]
    async fn l1_capacity_eviction_preserves_l2_entries() {
        const L1_CAPACITY: usize = 1024;
        let directory = tempfile::tempdir().unwrap();
        let cache = FoyerIndexCache::try_new(directory.path(), L1_CAPACITY, 4 * 1024 * 1024, 65536)
            .await
            .unwrap();
        let keys = (0..128_u8)
            .map(|value| InternalCacheKey::from_bytes([value; 16]))
            .collect::<Vec<_>>();
        for (value, key) in keys.iter().enumerate() {
            cache
                .insert(key, Arc::new(value as u64), 8, Some(test_codec()))
                .await;
        }
        cache.disk.storage().wait().await;
        assert!(cache.num_entries().await > 0);
        assert!(cache.num_entries().await < keys.len());
        assert!(cache.size_bytes().await <= L1_CAPACITY);

        // Find a real capacity eviction; don't manually clear or disable L1.
        let mut evicted = None;
        for (value, key) in keys.iter().enumerate() {
            if cache.state().memory.get(key, None).await.is_none() {
                evicted = Some((value as u64, key));
                break;
            }
        }
        let (expected, key) = evicted.expect("L1 budget must evict some entries");
        let (value, cached) = cache
            .get_or_insert(
                key,
                Box::pin(async { panic!("L1 eviction must not discard the L2 copy") }),
                Some(test_codec()),
            )
            .await
            .unwrap();
        assert!(cached);
        assert_eq!(*value.downcast_ref::<u64>().unwrap(), expected);
        assert_eq!(cache.stats.snapshot().disk_hits, 1);
        assert!(cache.state().memory.get(key, None).await.is_some());
        assert!(cache.size_bytes().await <= L1_CAPACITY);
    }

    #[tokio::test]
    async fn entries_larger_than_l1_capacity_remain_readable_from_disk() {
        let directory = tempfile::tempdir().unwrap();
        let key = InternalCacheKey::from_bytes([23; 16]);
        let codec = CacheCodec::new(
            "lance.test.Bytes",
            1,
            |value, writer| writer.write_raw(value.downcast_ref::<Vec<u8>>().unwrap()),
            |reader| Ok(Arc::new(reader.read_raw()?.to_vec())),
        );
        let value = vec![23_u8; 2048];
        let cache = FoyerIndexCache::try_new(directory.path(), 1024, 1024 * 1024, 65536)
            .await
            .unwrap();
        cache
            .insert(&key, Arc::new(value.clone()), value.len(), Some(codec))
            .await;
        assert_eq!(cache.num_entries().await, 0);
        cache.disk.close().await.unwrap();
        drop(cache);
        crate::foyer_cache::wait_for_directory_release(directory.path()).await;

        let cache = FoyerIndexCache::try_new(directory.path(), 1024, 1024 * 1024, 65536)
            .await
            .unwrap();
        for _ in 0..2 {
            let (entry, cached) = cache
                .get_or_insert(
                    &key,
                    Box::pin(async { panic!("an entry refused by L1 can still hit L2") }),
                    Some(codec),
                )
                .await
                .unwrap();
            assert!(cached);
            assert_eq!(entry.downcast_ref::<Vec<u8>>().unwrap(), &value);
            assert_eq!(cache.size_bytes().await, 0);
        }
        let stats = cache.stats.snapshot();
        assert_eq!((stats.memory_hits, stats.memory_misses), (0, 2));
        assert_eq!((stats.disk_hits, stats.disk_misses), (2, 0));
    }

    #[tokio::test]
    async fn shared_l2_capacity_eviction_preserves_l1_and_allows_source_refill() {
        let directory = tempfile::tempdir().unwrap();
        let cache = test_cache(directory.path()).await;
        let key = InternalCacheKey::from_bytes([24; 16]);
        let physical =
            FoyerIndexCache::disk_key_for_generation(&key, test_codec(), cache.state().generation);
        cache
            .insert(&key, Arc::new(42_u64), 8, Some(test_codec()))
            .await;
        cache.disk.storage().wait().await;
        assert!(cache.disk.get(&physical).await.unwrap().is_some());

        // Data and index entries compete for the same 1 MiB disk budget.
        // Drain each write so admission buffering cannot mask disk eviction.
        for block in 0..128 {
            cache.disk.insert(
                format!("lance-data-v1\0pressure\0{block}"),
                Bytes::from(vec![0_u8; 32768]),
            );
            cache.disk.storage().wait().await;
        }
        assert!(cache.disk.get(&physical).await.unwrap().is_none());
        let before = cache.stats.snapshot();
        let (entry, cached) = cache
            .get_or_insert(
                &key,
                Box::pin(async { panic!("L2 eviction must not invalidate L1") }),
                Some(test_codec()),
            )
            .await
            .unwrap();
        assert!(cached);
        assert_eq!(*entry.downcast_ref::<u64>().unwrap(), 42);
        assert_eq!(cache.stats.snapshot().disk_misses, before.disk_misses);

        cache.state().memory.clear().await;
        let loads = AtomicU64::new(0);
        for expected_cached in [false, true] {
            let (entry, cached) = cache
                .get_or_insert(
                    &key,
                    Box::pin(async {
                        loads.fetch_add(1, Ordering::Relaxed);
                        Ok((Arc::new(42_u64) as CacheEntry, 8))
                    }),
                    Some(test_codec()),
                )
                .await
                .unwrap();
            assert_eq!(cached, expected_cached);
            assert_eq!(*entry.downcast_ref::<u64>().unwrap(), 42);
        }
        assert_eq!(loads.load(Ordering::Relaxed), 1);
        assert_eq!(cache.stats.snapshot().disk_misses, before.disk_misses + 1);
        let allocated = fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| entry.unwrap())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("foyer-storage-direct-fs-")
            })
            .map(|entry| entry.metadata().unwrap().len())
            .sum::<u64>();
        assert!(allocated > 0 && allocated <= 1024 * 1024);
    }

    #[tokio::test]
    async fn concurrent_requests_share_one_source_load() {
        let directory = tempfile::tempdir().unwrap();
        let cache = test_cache(directory.path()).await;
        let key = InternalCacheKey::from_bytes([7; 16]);
        let loads = AtomicU64::new(0);
        let results = futures::future::join_all((0..8).map(|_| {
            cache.get_or_insert(
                &key,
                Box::pin(async {
                    loads.fetch_add(1, Ordering::Relaxed);
                    tokio::task::yield_now().await;
                    Ok((Arc::new(42_u64) as CacheEntry, 8))
                }),
                Some(test_codec()),
            )
        }))
        .await;
        assert_eq!(loads.load(Ordering::Relaxed), 1);
        let hits = results
            .into_iter()
            .map(|result| result.unwrap())
            .filter(|(_, hit)| *hit)
            .count();
        assert_eq!(hits, 7);
        let stats = cache.stats.snapshot();
        assert_eq!((stats.memory_hits, stats.memory_misses), (7, 1));
        assert_eq!((stats.disk_hits, stats.disk_misses), (0, 1));
    }

    #[tokio::test]
    async fn malformed_entries_miss_and_entries_without_codecs_stay_in_memory() {
        let directory = tempfile::tempdir().unwrap();
        let cache = test_cache(directory.path()).await;
        let key = InternalCacheKey::from_bytes([3; 16]);
        let physical =
            FoyerIndexCache::disk_key_for_generation(&key, test_codec(), cache.state().generation);
        cache.disk.insert(physical, Bytes::from_static(b"bad"));
        assert!(cache.get(&key, Some(test_codec())).await.is_none());
        assert_eq!(cache.stats.snapshot().decode_failures, 1);

        let before = cache.stats.snapshot();
        cache.insert(&key, Arc::new(9_u64), 8, None).await;
        assert!(cache.get(&key, None).await.is_some());
        cache.state().memory.clear().await;
        assert!(cache.get(&key, None).await.is_none());
        let after = cache.stats.snapshot();
        assert_eq!(after.disk_write_bytes, before.disk_write_bytes);
        assert_eq!(after.disk_misses, before.disk_misses);
    }

    #[tokio::test]
    async fn clear_isolates_in_flight_loads() {
        let directory = tempfile::tempdir().unwrap();
        let cache = Arc::new(test_cache(directory.path()).await);
        let key = InternalCacheKey::from_bytes([4; 16]);
        let (started_tx, started_rx) = oneshot::channel();
        let (resume_tx, resume_rx) = oneshot::channel();
        let loading = cache.clone();
        let task = tokio::spawn(async move {
            loading
                .get_or_insert(
                    &key,
                    Box::pin(async move {
                        started_tx.send(()).unwrap();
                        resume_rx.await.unwrap();
                        Ok((Arc::new(17_u64) as CacheEntry, 8))
                    }),
                    Some(test_codec()),
                )
                .await
                .unwrap()
        });
        started_rx.await.unwrap();
        cache.clear().await;
        cache
            .insert(&key, Arc::new(42_u64), 8, Some(test_codec()))
            .await;
        resume_tx.send(()).unwrap();
        assert_eq!(*task.await.unwrap().0.downcast_ref::<u64>().unwrap(), 17);
        assert_eq!(
            *cache
                .get(&key, Some(test_codec()))
                .await
                .unwrap()
                .downcast_ref::<u64>()
                .unwrap(),
            42
        );
        cache.state().memory.clear().await;
        assert_eq!(
            *cache
                .get(&key, Some(test_codec()))
                .await
                .unwrap()
                .downcast_ref::<u64>()
                .unwrap(),
            42
        );
    }

    #[tokio::test]
    async fn clear_invalidates_even_if_generation_cannot_be_persisted() {
        let directory = tempfile::tempdir().unwrap();
        let mut cache = test_cache(directory.path()).await;
        let key = InternalCacheKey::from_bytes([5; 16]);
        cache
            .insert(&key, Arc::new(7_u64), 8, Some(test_codec()))
            .await;
        cache.generation_path = directory.path().join("missing-parent/generation");
        cache.clear().await;
        assert!(cache.get(&key, Some(test_codec())).await.is_none());
    }

    #[tokio::test]
    async fn recovers_index_entries_and_persisted_invalidation() {
        let directory = tempfile::tempdir().unwrap();
        let key = InternalCacheKey::from_bytes([6; 16]);
        let cache = test_cache(directory.path()).await;
        cache
            .insert(&key, Arc::new(99_u64), 8, Some(test_codec()))
            .await;
        cache.disk.close().await.unwrap();
        drop(cache);
        crate::foyer_cache::wait_for_directory_release(directory.path()).await;

        let recovered = test_cache(directory.path()).await;
        assert_eq!(
            *recovered
                .get(&key, Some(test_codec()))
                .await
                .unwrap()
                .downcast_ref::<u64>()
                .unwrap(),
            99
        );
        assert_eq!(recovered.stats.snapshot().disk_hits, 1);
        recovered.clear().await;
        recovered.disk.close().await.unwrap();
        drop(recovered);
        crate::foyer_cache::wait_for_directory_release(directory.path()).await;

        let cleared = test_cache(directory.path()).await;
        assert!(cleared.get(&key, Some(test_codec())).await.is_none());
    }

    #[test]
    fn corrupt_generation_starts_a_new_namespace() {
        let directory = tempfile::tempdir().unwrap();
        let path = generation_path(directory.path());
        fs::write(&path, "truncated").unwrap();
        let generation = load_generation(directory.path()).unwrap();
        assert_eq!(load_generation(directory.path()).unwrap(), generation);
        assert_eq!(fs::read_to_string(path).unwrap(), generation.to_string());
    }

    #[test]
    fn codec_type_is_part_of_the_physical_key() {
        let key =
            InternalCacheKey::from_bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let codec = CacheCodec::new(
            "lance.test.IndexEntry",
            1,
            |_value, _writer| Ok(()),
            |_reader| unreachable!(),
        );
        let physical_key = FoyerIndexCache::disk_key(&key, codec);
        assert!(physical_key.starts_with("lance-index-v1:blake3-128-v1:0:lance.test.IndexEntry:"));
        assert!(!physical_key.contains('\0'));
        assert_ne!(
            physical_key,
            FoyerIndexCache::disk_key_for_generation(&key, codec, 1)
        );
    }

    #[tokio::test]
    async fn clear_does_not_clear_shared_data_entries() {
        let directory = tempfile::tempdir().unwrap();
        let disk = crate::foyer_cache::build_disk_cache(directory.path(), 1024 * 1024, 64 * 1024)
            .await
            .unwrap();
        disk.insert(
            "lance-data-v1\0test".to_owned(),
            Bytes::from_static(b"data-entry"),
        );
        let index =
            FoyerIndexCache::from_cache(disk.clone(), 1024 * 1024, directory.path()).unwrap();

        index.clear().await;

        assert!(disk.get("lance-data-v1\0test").await.unwrap().is_some());
    }
}
