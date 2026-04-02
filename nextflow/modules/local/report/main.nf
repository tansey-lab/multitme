process REPORT {
    tag "$meta.id"
    label 'process_medium'

    container "${params.container_registry}/multitme:${params.container_version}"

    input:
    tuple val(meta), path(predictions_h5ad)
    tuple val(meta), path(probs_npy)
    tuple val(meta), path(latent_npy)

    output:
    tuple val(meta), path("${meta.id}_cell_type_summary.pdf"),     emit: summary_pdf
    tuple val(meta), path("${meta.id}_confidence_analysis.pdf"),   emit: confidence_pdf
    tuple val(meta), path("${meta.id}_spatial_plots.pdf"),         emit: spatial_pdf
    tuple val(meta), path("${meta.id}_choropleth.html"),           emit: choropleth
    path "versions.yml",                                            emit: versions

    script:
    def args = task.ext.args ?: ''
    """
    multitme-report \\
        --input ${predictions_h5ad} \\
        --probs ${probs_npy} \\
        --latent ${latent_npy} \\
        --output-dir . \\
        ${args}

    mv cell_type_summary.pdf ${meta.id}_cell_type_summary.pdf
    mv confidence_analysis.pdf ${meta.id}_confidence_analysis.pdf
    mv spatial_plots.pdf ${meta.id}_spatial_plots.pdf
    mv choropleth.html ${meta.id}_choropleth.html

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        multitme: \$(python3 -c "import multitme; print(multitme.__version__)")
        matplotlib: \$(python3 -c "import matplotlib; print(matplotlib.__version__)")
    END_VERSIONS
    """
}
