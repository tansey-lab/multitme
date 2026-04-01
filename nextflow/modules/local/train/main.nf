process TRAIN {
    tag "$meta.id"
    label 'process_gpu'

    container "ghcr.io/${params.container_registry}/multitme:${params.container_version}"

    input:
    tuple val(meta), path(config)
    tuple val(meta), path(scrna_h5ad)
    tuple val(meta), path(xenium_h5ad)

    output:
    tuple val(meta), path("${meta.id}_model.pt"),    emit: model
    tuple val(meta), path("${meta.id}_metadata.pt"), emit: metadata
    path "versions.yml",                              emit: versions

    script:
    def args = task.ext.args ?: ''
    """
    multitme-train \\
        --config ${config} \\
        data.scrna_path=${scrna_h5ad} \\
        data.xenium_path=${xenium_h5ad} \\
        output.dir=. \\
        ${args}

    mv model.pt ${meta.id}_model.pt
    mv metadata.pt ${meta.id}_metadata.pt

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        multitme: \$(python3 -c "import multitme; print(multitme.__version__)")
        pytorch: \$(python3 -c "import torch; print(torch.__version__)")
    END_VERSIONS
    """
}
