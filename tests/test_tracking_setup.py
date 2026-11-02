def test_tracking_local_environment_defines_both_ltr_and_tracker_settings(tmp_path):
    from reproduce import assets
    assert hasattr(assets,'write_tracking_environment'),'LTR legacy loader environment bootstrap is missing'
    source=tmp_path/'source';assets.write_tracking_environment(source,tmp_path/'assets')
    ltr={};exec((source/'ltr/admin/local.py').read_text(),ltr)
    settings=ltr['EnvironmentSettings']()
    assert settings.workspace_dir==str(tmp_path/'assets')
    assert (source/'pytracking/evaluation/local.py').is_file()

def test_removed_torchvision_model_urls_is_replaced_with_same_pretrained_family():
    from reproduce import assets
    assert hasattr(assets,'modernize_tracking_python'),'legacy Torchvision backbone imports are not adapted'
    raw='from torchvision.models.resnet import model_urls\nvalue=model_urls["resnet50"]\n'
    fixed=assets.modernize_tracking_python(raw)
    assert 'import model_urls' not in fixed and 'ResNet50_Weights.IMAGENET1K_V1.url' in fixed
    namespace={};exec(fixed,namespace)
    assert namespace['value'].startswith('https://download.pytorch.org/models/resnet50-')

def test_github_zip_roi_source_links_are_materialized_before_native_compilation(tmp_path):
    from reproduce import assets
    assert hasattr(assets,'materialize_prroi_links'),'GitHub ZIP symlink sources are not materialized'
    root=tmp_path/'roi';(root/'src').mkdir(parents=True)
    (root/'src/kernel.cu').write_text('int kernel_value = 1;\n')
    (root/'src/kernel.cuh').write_text('typedef float value_type;\n')
    link=root/'pytorch/prroi_pool/src/kernel.cu';link.parent.mkdir(parents=True)
    link.write_text('../../../src/kernel.cu')
    header=link.with_suffix('.cuh');header.write_text('../../../src/kernel.cuh')
    changed=assets.materialize_prroi_links(root)
    assert len(changed)==2 and link.read_text()=='int kernel_value = 1;\n'
    assert header.read_text()=='typedef float value_type;\n'
