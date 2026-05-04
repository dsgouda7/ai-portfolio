output "eks_cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = aws_eks_cluster.ml_cluster.endpoint
  sensitive   = false
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.ml_cluster.name
}

output "eks_cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.ml_cluster.arn
}

output "s3_artifacts_bucket_name" {
  description = "S3 bucket name for ML artifacts"
  value       = aws_s3_bucket.ml_artifacts.bucket
}

output "s3_artifacts_bucket_arn" {
  description = "S3 bucket ARN for ML artifacts"
  value       = aws_s3_bucket.ml_artifacts.arn
}

output "redis_endpoint" {
  description = "Redis cluster endpoint for feature store"
  value       = "${aws_elasticache_cluster.feature_store.cache_nodes[0].address}:${aws_elasticache_cluster.feature_store.cache_nodes[0].port}"
}

output "mlflow_database_endpoint" {
  description = "PostgreSQL database endpoint for MLflow"
  value       = aws_db_instance.mlflow_db.endpoint
  sensitive   = true
}

output "vpc_id" {
  description = "VPC ID for ML infrastructure"
  value       = aws_vpc.ml_vpc.id
}

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${aws_eks_cluster.ml_cluster.name}"
}
