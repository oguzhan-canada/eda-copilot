# ─────────────────────────────────────────────────────────────────────────────
# outputs.tf — Key information after terraform apply
# ─────────────────────────────────────────────────────────────────────────────

output "s3_bucket" {
  description = "S3 bucket for data handoff and checkpoints"
  value       = aws_s3_bucket.data.bucket
}

output "ami_id" {
  description = "Deep Learning AMI used for both instances"
  value       = local.ami_id
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "subnet_id" {
  description = "Public subnet ID"
  value       = aws_subnet.public.id
}

output "security_group_id" {
  description = "Security group ID for instances"
  value       = aws_security_group.instance.id
}

output "cpu_launch_template_id" {
  description = "Launch template ID for CPU instance"
  value       = aws_launch_template.cpu.id
}

output "gpu_launch_template_id" {
  description = "Launch template ID for GPU instance"
  value       = aws_launch_template.gpu.id
}

output "cpu_instance_public_ip" {
  description = "Public IP of CPU Spot instance (if launched)"
  value       = var.launch_cpu ? aws_spot_instance_request.cpu[0].public_ip : "not launched"
}

output "gpu_instance_public_ip" {
  description = "Public IP of GPU Spot instance (if launched)"
  value       = var.launch_gpu ? aws_spot_instance_request.gpu[0].public_ip : "not launched"
}

output "iam_instance_profile_arn" {
  description = "IAM instance profile ARN"
  value       = aws_iam_instance_profile.instance.arn
}
