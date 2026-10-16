"""Verify the completed expanded RGB cache using fixed development subjects only."""


def main():
    import io
    import json
    import time
    import zipfile
    from pathlib import Path
    import numpy as np
    from audit_mri import load_labels
    from cache_mri_rgb import file_checks, pose_features
    from cache_mri_robust_rgb import CONDITIONS, NOISE_SEED, ROOT, digest_json, save_json
    p = ROOT / 'artifacts/fusion_rgb_expanded_20261013'
    audit = json.loads((p / 'rgb_audit.json').read_text())
    protocol = json.loads((p / 'protocol.json').read_text())
    assert audit['status'] == 'complete'
    expected = protocol['train_subjects'] + protocol['validation_subjects']
    assert len(expected) == 16 and len(set(expected)) == 16
    assert set(row['subject'] for row in audit['subjects']) == set(expected)
    assert not audit['test_subjects_used']
    assert len(list(p.glob('subject*.npz'))) == 16
    summary = {'status': 'complete', 'actual_frames': 0, 'training_frames': 0, 'validation_frames': 0,
               'original64_frames_exact_reuse': 0, 'additional_actual_frames': 0,
               'new_inference_count': 0, 'test_subjects_used': [],
               'protocol_sha256': digest_json(protocol), 'source_subject_count': 16,
               'archive_authentication': protocol['archive_authentication'],
               'checks': 'All cache file hashes; prior64 poses/features/timestamps exact; frame IDs unique/sorted; actual decode next positions; noise seed; four pixel fingerprints; four features/availability reconstructed from saved raw poses and source focal; all missing retained',
               'conditions': {group: {c: {'frames': 0, 'root_available_frames': 0,
                                 'missing_frames': 0, 'raw_any_visible_frames': 0,
                                 'root_missing_but_partial_raw_frames': 0,
                                 'partial_visible_joint_total': 0} for c in CONDITIONS}
                              for group in ('train', 'validation', 'all')},
               'subjects': []}
    with zipfile.ZipFile(ROOT / 'dataset_release.zip') as archive:
        for row in audit['subjects']:
            subject = row['subject']
            path = p / ('subject' + str(subject) + '.npz')
            assert file_checks(path)[0] == row['cache_sha256']
            old_path = Path(protocol['old_cache_directory']) / path.name
            assert file_checks(old_path)[0] == row['old_cache_sha256']
            labels = load_labels(archive.read('dataset_release/aligned_data/pose_labels/subject' + str(subject) + '_all_labels.cpl'))
            focal = np.diag(labels['camera_matrix']['cam1_intrinsic'])[:2]
            with np.load(path, allow_pickle=False) as cache, np.load(old_path, allow_pickle=False) as old:
                ids = cache['frame_ids']
                reused = cache['reused_clean_dark']
                assert ids.shape == (96,) and np.all(np.diff(ids) > 0)
                assert reused.dtype == np.bool_ and int(reused.sum()) == 64
                np.testing.assert_array_equal(ids[reused], old['frame_ids'])
                np.testing.assert_array_equal(cache['timestamps_ms'][reused], old['timestamps_ms'])
                np.testing.assert_array_equal(cache['decode_next_frame_positions'], ids + 1)
                np.testing.assert_array_equal(cache['pixel_noise_seed_words'][:, 0], np.full(96, NOISE_SEED))
                np.testing.assert_array_equal(cache['pixel_noise_seed_words'][:, 1], np.full(96, subject))
                np.testing.assert_array_equal(cache['pixel_noise_seed_words'][:, 2], ids)
                assert str(cache['source_sha256']) == str(old['source_sha256']) == row['source_sha256']
                assert int(cache['source_crc32']) == row['source_crc32']
                assert str(cache['protocol_sha256']) == summary['protocol_sha256']
                group = 'train' if subject in protocol['train_subjects'] else 'validation'
                summary['actual_frames'] += 96
                summary['training_frames' if group == 'train' else 'validation_frames'] += 96
                summary['original64_frames_exact_reuse'] += 64
                summary['additional_actual_frames'] += 32
                for c in CONDITIONS:
                    poses = cache[c + '_poses']
                    features, available = pose_features(poses, focal)
                    np.testing.assert_array_equal(features, cache[c + '_features'])
                    np.testing.assert_array_equal(available, cache[c + '_available'])
                    np.testing.assert_array_equal(~available[:, 11], cache[c + '_missing'])
                    assert len(cache[c + '_pixel_sha256']) == 96
                    assert all(len(str(v)) == 64 and all(x in '0123456789abcdef' for x in str(v)) for v in cache[c + '_pixel_sha256'])
                    if c in ('clean', 'dark'):
                        for key in ('poses', 'features'):
                            np.testing.assert_array_equal(cache[c + '_' + key][reused], old[c + '_' + key])
                    visible = poses[..., 2] >= .3
                    partial = cache[c + '_missing'] & visible.any(-1)
                    assert np.all(cache[c + '_features'][partial] == 0)
                    summary['new_inference_count'] += row['conditions'][c]['new_inference_count']
                    values = {'frames': 96, 'root_available_frames': int(available[:, 11].sum()),
                              'missing_frames': int((~available[:, 11]).sum()),
                              'raw_any_visible_frames': int(visible.any(-1).sum()),
                              'root_missing_but_partial_raw_frames': int(partial.sum()),
                              'partial_visible_joint_total': int(visible[partial].sum())}
                    for category in (group, 'all'):
                        for key, value in values.items():
                            summary['conditions'][category][c][key] += value
                summary['subjects'].append({'subject': subject, 'cache_sha256': row['cache_sha256'], 'status': 'verified'})
    assert summary['actual_frames'] == 1536
    assert summary['training_frames'] == 1152 and summary['validation_frames'] == 384
    assert summary['original64_frames_exact_reuse'] == 1024 and summary['additional_actual_frames'] == 512
    assert summary['new_inference_count'] == 4096
    summary['cache_generation_elapsed_min'] = (audit['updated_unix'] - audit['started_unix']) / 60
    summary['verified_unix'] = time.time()
    save_json(p / 'cache_verification.json', summary)
    print(json.dumps(summary, indent=2))



if __name__ == "__main__":
    main()

