process TRAIN {
    tag "$meta.id"
    label 'process_gpu'

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
    """
    multitme-train \\
        --config ${config} \\
        data.scrna_path=${scrna} \\
        data.xenium_path=${xenium} \\
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
