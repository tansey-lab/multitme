process TRAIN {
    tag "$meta.id"
    label 'process_gpu'
    label 'process_high'

    container "${params.container_registry}/multitme:${params.container_version}"

    input:
    tuple val(meta), path(config)
    tuple val(meta), path(scrna)
    tuple val(meta), path(xenium)
    tuple val(meta), path(scrna_preprocessed)
    tuple val(meta), path(xenium_preprocessed)

    output:
    tuple val(meta), path("${meta.id}_checkpoint.pt"), emit: checkpoint
    path "versions.yml",                                emit: versions

    script:
    def args = task.ext.args ?: ''
    def wandb_key = params.wandb_api_key ?: ''
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
        --scrna-preprocessed ${scrna_preprocessed} \\
        --xenium-preprocessed ${xenium_preprocessed} \\
        data.annotation_column=${params.annotation_column ?: 'major_annotation'} \\
        output.dir=. \\
        ${args}

    mv checkpoint.pt ${meta.id}_checkpoint.pt

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        multitme: \$(python3 -c "import multitme; print(multitme.__version__)")
        pytorch: \$(python3 -c "import torch; print(torch.__version__)")
    END_VERSIONS
    """
}
