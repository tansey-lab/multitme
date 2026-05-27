process TRAIN {
    tag "$meta.id"
    label 'process_gpu'
    label 'process_high'

    container "${params.container_registry}/multitme:${params.container_version}"

    input:
    tuple val(meta), path(config)
    tuple val(meta), path(scrna)
    tuple val(meta), path(xenium)

    output:
    tuple val(meta), path("${meta.id}_checkpoint.pt"), emit: checkpoint
    path "versions.yml",                                emit: versions

    script:
    def args = task.ext.args ?: ''
    def wandb_key = params.wandb_api_key ?: ''
    def spatial_overrides = (params.spatial_enabled ?: false) ? (
        "spatial.enabled=true " +
        "spatial.obsm_key=${params.spatial_obsm_key ?: 'spatial'} " +
        "spatial.tile_size=${params.spatial_tile_size ?: 500.0} " +
        "spatial.halo=${params.spatial_halo ?: 150.0} " +
        "spatial.weight=${params.spatial_weight ?: 1.0}"
    ) : ''
    """
    # Set wandb API key if provided
    if [ -n "${wandb_key}" ]; then
        export WANDB_API_KEY="${wandb_key}"
    fi

    # Use consistent run ID across retries so wandb resumes to same run
    export WANDB_RUN_ID="${workflow.runName}_${meta.id}"
    export WANDB_RESUME="allow"

    multitme-train \\
        --config ${config} \\
        --scrna ${scrna} \\
        --xenium ${xenium} \\
        data.annotation_column=${params.annotation_column ?: 'major_annotation'} \\
        output.dir=. \\
        ${spatial_overrides} \\
        ${args}

    mv checkpoint.pt ${meta.id}_checkpoint.pt

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        multitme: \$(python3 -c "import multitme; print(multitme.__version__)")
        pytorch: \$(python3 -c "import torch; print(torch.__version__)")
    END_VERSIONS
    """
}
