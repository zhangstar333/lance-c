// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared L2 Foyer disk cache for data-file blocks and serialized index entries.
//! L1 is the Session's index and metadata memory cache; Foyer adds no L1 capacity.

use std::ffi::c_char;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use foyer::{
    BlockEngineConfig, DeviceBuilder, EventListener, FsDeviceBuilder, HybridCache,
    HybridCacheBuilder, HybridCachePolicy, PsyncIoEngineConfig,
};

use crate::error::ffi_try;
use crate::foyer_data_cache::FoyerDataCache;
use crate::foyer_index_cache::FoyerIndexCache;
use crate::helpers;
use crate::runtime::block_on;
use crate::session::{LanceSession, session_new_with_factories, u64_to_usize};

pub(crate) const PAGE_SIZE: usize = 4096;
pub(crate) const DEFAULT_READ_BLOCK_SIZE: usize = 1024 * 1024;
pub(crate) const DEFAULT_STORAGE_BLOCK_SIZE: usize = 16 * 1024 * 1024;

/// One disk path and total capacity shared by data blocks and index entries.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LanceFoyerCacheOptions {
    pub directory: *const c_char,
    pub disk_capacity_bytes: u64,
}

/// Create a session with a shared Foyer data/index disk cache.
/// NULL options disable the disk cache. Block sizes are managed internally.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_session_new_with_foyer_cache(
    index_cache_size_bytes: u64,
    metadata_cache_size_bytes: u64,
    foyer_cache_options: *const LanceFoyerCacheOptions,
) -> *mut LanceSession {
    ffi_try!(
        unsafe {
            session_new_with_foyer_cache_inner(
                index_cache_size_bytes,
                metadata_cache_size_bytes,
                foyer_cache_options,
            )
        },
        null
    )
}

unsafe fn session_new_with_foyer_cache_inner(
    index_cache_size_bytes: u64,
    metadata_cache_size_bytes: u64,
    foyer_cache_options: *const LanceFoyerCacheOptions,
) -> lance_core::Result<*mut LanceSession> {
    let memory_capacity = u64_to_usize(index_cache_size_bytes, "index_cache_size_bytes")?;
    u64_to_usize(metadata_cache_size_bytes, "metadata_cache_size_bytes")?;
    let Some(options) = (unsafe { foyer_cache_options.as_ref() }) else {
        return session_new_with_factories(
            index_cache_size_bytes,
            metadata_cache_size_bytes,
            None,
            None,
            None,
        );
    };
    let directory = unsafe { helpers::parse_c_string(options.directory)? }.ok_or_else(|| {
        lance_core::Error::invalid_input_source(
            "foyer_cache_options.directory must not be NULL".into(),
        )
    })?;
    if directory.is_empty() {
        return Err(lance_core::Error::invalid_input_source(
            "foyer_cache_options.directory must not be empty".into(),
        ));
    }
    let disk_capacity = u64_to_usize(options.disk_capacity_bytes, "disk_capacity_bytes")?;
    validate_disk_cache_sizes(disk_capacity, DEFAULT_STORAGE_BLOCK_SIZE)?;
    let disk = block_on(build_disk_cache(
        Path::new(directory),
        disk_capacity,
        DEFAULT_STORAGE_BLOCK_SIZE,
    ))
    .map_err(|error| {
        lance_core::Error::io(format!(
            "failed to initialize Foyer cache at {directory:?}: {error}",
        ))
    })?;
    let data_cache = FoyerDataCache::from_cache(disk.clone(), DEFAULT_READ_BLOCK_SIZE);
    let index_cache = FoyerIndexCache::from_cache(disk, memory_capacity, Path::new(directory))
        .map_err(|error| {
            lance_core::Error::io(format!(
                "failed to initialize index cache generation at {directory:?}: {error}",
            ))
        })?;
    let index_stats = index_cache.stats();
    session_new_with_factories(
        index_cache_size_bytes,
        metadata_cache_size_bytes,
        Some(Arc::new(index_cache)),
        Some(index_stats),
        Some(Arc::new(data_cache)),
    )
}

pub(crate) fn validate_disk_cache_sizes(
    disk_capacity: usize,
    storage_block_size: usize,
) -> lance_core::Result<()> {
    // A storage block includes a 4 KiB blob index. Reserve blocks for the
    // flusher and reclaimer as well as retained entries.
    if storage_block_size <= PAGE_SIZE || !storage_block_size.is_multiple_of(PAGE_SIZE) {
        return Err(lance_core::Error::invalid_input_source(format!(
            "storage_block_size_bytes={storage_block_size} must be a multiple of {PAGE_SIZE} and greater than {PAGE_SIZE}"
        ).into()));
    }
    let minimum_capacity = storage_block_size.checked_mul(4).ok_or_else(|| {
        lance_core::Error::invalid_input_source(
            format!("storage_block_size_bytes={storage_block_size} is too large").into(),
        )
    })?;
    if disk_capacity < minimum_capacity || !disk_capacity.is_multiple_of(PAGE_SIZE) {
        return Err(lance_core::Error::invalid_input_source(format!(
            "disk_capacity_bytes={disk_capacity} must be a multiple of {PAGE_SIZE} and at least {minimum_capacity} (four storage blocks of {storage_block_size} bytes)"
        ).into()));
    }
    Ok(())
}

// Foyer does not lock its FsDevice directory. Its event listener also stays
// alive while HybridCache's asynchronous drop drains storage, keeping the
// directory locked until pending background I/O completes.
struct CacheDirectory {
    lock: File,
}

impl Drop for CacheDirectory {
    fn drop(&mut self) {
        // Closing our descriptor alone can leave the lock held by a copy
        // inherited during a concurrent process spawn. Release the lock when
        // the last cache owner finishes, including initialization failures.
        if let Err(error) = self.lock.unlock() {
            log::warn!("failed to release Foyer cache directory lock: {error}");
        }
    }
}

impl EventListener for CacheDirectory {
    type Key = String;
    type Value = Bytes;
}

fn lock_directory(
    directory: &Path,
    disk_capacity: usize,
    storage_block_size: usize,
) -> std::result::Result<Arc<CacheDirectory>, foyer::Error> {
    fs::create_dir_all(directory).map_err(foyer::Error::io_error)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(directory.join("lance-cache-layout"))
        .map_err(foyer::Error::io_error)?;
    lock.try_lock().map_err(|error| {
        foyer::Error::new(foyer::ErrorKind::Config, format!(
            "cannot exclusively lock cache directory {directory:?}: {error}; share one LanceSession per directory"
        ))
    })?;
    let mut directory_guard = CacheDirectory { lock };
    let lock = &mut directory_guard.lock;
    let expected = format!("lance-disk-v1 {disk_capacity} {storage_block_size}\n");
    let mut layout = String::new();
    lock.read_to_string(&mut layout)
        .map_err(foyer::Error::io_error)?;
    if layout.is_empty() {
        // Adopt an older cache only if opening it will not truncate existing
        // partitions or leave files outside the requested capacity.
        let mut partitions = 0;
        for entry in fs::read_dir(directory).map_err(foyer::Error::io_error)? {
            let entry = entry.map_err(foyer::Error::io_error)?;
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with("foyer-storage-direct-fs-")
            {
                let size = entry.metadata().map_err(foyer::Error::io_error)?.len();
                if size != storage_block_size as u64 {
                    return Err(foyer::Error::new(
                        foyer::ErrorKind::Config,
                        format!(
                            "cache partition {:?} has size {size}, expected {storage_block_size}; use a new cache directory",
                            entry.path()
                        ),
                    ));
                }
                partitions += 1;
            }
        }
        if partitions > disk_capacity / storage_block_size {
            return Err(foyer::Error::new(
                foyer::ErrorKind::Config,
                format!(
                    "cache directory {directory:?} has {partitions} partitions exceeding disk_capacity_bytes={disk_capacity}; use a new cache directory"
                ),
            ));
        }
        lock.write_all(expected.as_bytes())
            .map_err(foyer::Error::io_error)?;
        lock.sync_all().map_err(foyer::Error::io_error)?;
    } else if layout != expected {
        return Err(foyer::Error::new(
            foyer::ErrorKind::Config,
            format!(
                "cache directory {directory:?} has layout {layout:?}, requested {expected:?}; use a new cache directory when changing disk capacity or storage block size"
            ),
        ));
    }
    Ok(Arc::new(directory_guard))
}

/// Build a disk-only Foyer cache. The caller supplies logical namespaces in
/// its keys, allowing data-file and index entries to share one capacity pool
/// without colliding.
pub(crate) async fn build_disk_cache(
    directory: &Path,
    disk_capacity: usize,
    storage_block_size: usize,
) -> std::result::Result<HybridCache<String, Bytes>, foyer::Error> {
    validate_disk_cache_sizes(disk_capacity, storage_block_size)
        .map_err(|error| foyer::Error::new(foyer::ErrorKind::Config, error.to_string()))?;
    let directory_lock = lock_directory(directory, disk_capacity, storage_block_size)?;
    let device = FsDeviceBuilder::new(directory)
        .with_capacity(disk_capacity)
        .build()?;
    let engine = BlockEngineConfig::new(device).with_block_size(storage_block_size);
    HybridCacheBuilder::new()
        .with_name("lance_disk")
        .with_event_listener(directory_lock)
        .with_policy(HybridCachePolicy::WriteOnInsertion)
        .with_flush_on_close(false)
        .memory(0)
        .with_shards(1)
        .with_weighter(|_key: &String, value: &Bytes| value.len().max(1))
        .storage()
        .with_io_engine_config(PsyncIoEngineConfig::new())
        .with_engine_config(engine)
        .build()
        .await
}

#[cfg(test)]
pub(crate) async fn wait_for_directory_release(directory: &Path) {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(directory.join("lance-cache-layout"))
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            match file.try_lock() {
                Ok(()) => {
                    file.unlock().unwrap();
                    return;
                }
                Err(std::fs::TryLockError::WouldBlock) => {
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
                Err(error) => panic!("cannot lock closed cache directory: {error}"),
            }
        }
    })
    .await
    .expect("cache directory remained locked after all owners closed");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unusable_storage_geometry() {
        assert!(validate_disk_cache_sizes(1024 * 1024, 4096).is_err());
        assert!(validate_disk_cache_sizes(3 * 65536, 65536).is_err());
        assert!(validate_disk_cache_sizes(4 * 65536, 65536).is_ok());
        assert!(validate_disk_cache_sizes(4 * 65536 + 1, 65536).is_err());
    }

    #[tokio::test]
    async fn directory_lock_follows_shared_cache_lifetime() {
        let directory = tempfile::tempdir().unwrap();
        let cache = build_disk_cache(directory.path(), 1024 * 1024, 65536)
            .await
            .unwrap();
        let shared = cache.clone();
        drop(cache);
        assert!(
            build_disk_cache(directory.path(), 1024 * 1024, 65536)
                .await
                .is_err()
        );
        shared.insert("test".to_owned(), Bytes::from_static(b"cached"));
        shared.close().await.unwrap();
        drop(shared);
        wait_for_directory_release(directory.path()).await;
        let reopened = build_disk_cache(directory.path(), 1024 * 1024, 65536)
            .await
            .unwrap();
        assert_eq!(
            reopened
                .get("test")
                .await
                .unwrap()
                .unwrap()
                .value()
                .as_ref(),
            b"cached"
        );
        reopened.close().await.unwrap();
    }

    #[test]
    fn directory_guard_releases_lock_even_if_descriptor_was_duplicated() {
        let directory = tempfile::tempdir().unwrap();
        let guard = lock_directory(directory.path(), 1024 * 1024, 65536).unwrap();
        // A concurrent process spawn can temporarily inherit the same open
        // file description. Closing only our descriptor would retain its lock.
        let inherited = guard.lock.try_clone().unwrap();
        drop(guard);
        let reopened = lock_directory(directory.path(), 1024 * 1024, 65536).unwrap();
        drop(inherited);
        drop(reopened);
    }

    #[test]
    fn rejects_layout_changes_and_legacy_partition_truncation() {
        let directory = tempfile::tempdir().unwrap();
        let guard = lock_directory(directory.path(), 1024 * 1024, 65536).unwrap();
        drop(guard);
        assert!(lock_directory(directory.path(), 2 * 1024 * 1024, 65536).is_err());
        assert!(lock_directory(directory.path(), 1024 * 1024, 131072).is_err());
        lock_directory(directory.path(), 1024 * 1024, 65536).unwrap();

        let legacy = tempfile::tempdir().unwrap();
        let partition = legacy.path().join("foyer-storage-direct-fs-00000000");
        File::create(&partition).unwrap().set_len(65536).unwrap();
        assert!(lock_directory(legacy.path(), 1024 * 1024, 131072).is_err());
        assert_eq!(fs::metadata(&partition).unwrap().len(), 65536);
        lock_directory(legacy.path(), 1024 * 1024, 65536).unwrap();
    }
}
