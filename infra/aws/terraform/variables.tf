# ─────────────────────────────────────────────────────────────────────────────
# variables.tf — All configurable parameters for the EDA-KG AWS deployment
# ─────────────────────────────────────────────────────────────────────────────

variable "aws_region" {
  description = "AWS region to deploy in"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project prefix for all resource names"
  type        = string
  default     = "eda-kg"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into instances (your IP/32)"
  type        = string
  default     = "0.0.0.0/0"
}

# ── Instance Types (with fallbacks for Spot capacity) ─────────────────────────

variable "cpu_instance_types" {
  description = "Ordered list of CPU instance types for Stages 1-2"
  type        = list(string)
  default     = ["c6i.4xlarge", "c6i.8xlarge", "c6a.4xlarge"]
}

variable "gpu_instance_types" {
  description = "Ordered list of GPU instance types for Stages 3-5"
  type        = list(string)
  default     = ["g5.2xlarge", "g4dn.xlarge"]
}

# ── Storage ───────────────────────────────────────────────────────────────────

variable "cpu_ebs_size_gb" {
  description = "Root EBS volume size for CPU instance (GB)"
  type        = number
  default     = 200
}

variable "gpu_ebs_size_gb" {
  description = "Root EBS volume size for GPU instance (GB)"
  type        = number
  default     = 100
}

variable "s3_bucket_name" {
  description = "S3 bucket name for data handoff (must be globally unique)"
  type        = string
  default     = ""
}

# ── Docker / ECR (removed — using direct install on DLAMI) ────────────────────
# ECR no longer needed. OpenROAD installed from source, Python deps via pip.

# ── Spot ──────────────────────────────────────────────────────────────────────

variable "spot_max_price_cpu" {
  description = "Max Spot price for CPU instance ($/hr). Empty = on-demand cap."
  type        = string
  default     = "0.60"
}

variable "spot_max_price_gpu" {
  description = "Max Spot price for GPU instance ($/hr). Empty = on-demand cap."
  type        = string
  default     = "0.35"
}

# ── Stage Control ─────────────────────────────────────────────────────────────

variable "launch_cpu" {
  description = "Set to true to launch the CPU Spot instance (Stages 1-2)"
  type        = bool
  default     = false
}

variable "launch_gpu" {
  description = "Set to true to launch the GPU Spot instance (Stages 3-5)"
  type        = bool
  default     = false
}

# ── AMI ───────────────────────────────────────────────────────────────────────

variable "ami_id" {
  description = "AMI ID for both instances. Empty = latest AWS Deep Learning AMI (Ubuntu 22.04)."
  type        = string
  default     = ""
}

# ── High Memory Designs ───────────────────────────────────────────────────────

variable "high_memory_designs" {
  description = "Designs requiring r6i.2xlarge (64 GB) — swerv_wrapper OOMs on 32 GB"
  type        = list(string)
  default     = ["swerv_wrapper"]
}

variable "high_memory_instance_types" {
  description = "Instance types for high-memory designs"
  type        = list(string)
  default     = ["r6i.2xlarge", "r6a.2xlarge", "m6i.4xlarge"]
}
