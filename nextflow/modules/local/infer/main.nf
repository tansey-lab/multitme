process INFER {
    tag "$meta.id"
    label 'process_gpu'

    container "${params.container_registry}/multitme:${params.container_version}"

    input:
    tuple val(meta), path(checkpoint)
    tuple val(meta), path(xenium)
    tuple val(meta), path(scrna)

    output:
    tuple val(meta), path("${meta.id}_predictions.h5ad"), emit: predictions
    tuple val(meta), path("${meta.id}_latent.npz"),       emit: latent
    tuple val(meta), path("${meta.id}_pred_probs.npy"),   emit: probs
    path "versions.yml",                                   emit: versions

    script:
    def modality = meta.modality ?: 'xenium'
    def args = task.ext.args ?: ''
    """
    multitme-infer \\
        --checkpoint ${checkpoint} \\
        --input ${xenium} \\
        --scrna ${scrna} \\
        --modality ${modality} \\
        --output-dir . \\
        ${args}

    mv predictions.h5ad ${meta.id}_predictions.h5ad
    mv latent.npz ${meta.id}_latent.npz
    mv pred_probs.npy ${meta.id}_pred_probs.npy

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        multitme: \$(python3 -c "import multitme; print(multitme.__version__)")
    END_VERSIONS
    """
}
