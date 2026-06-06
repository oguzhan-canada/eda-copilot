# ─────────────────────────────────────────────────────────────────────────────
# main.tf — EDA-KG AWS Infrastructure
#
# Two-instance Spot strategy with S3 bridge:
#   Stage 1-2: c6i.8xlarge (CPU) for OpenROAD EDA + feature extraction
#   Stage 3-5: g4dn.xlarge (GPU) for training + optimization + evaluation
#
# Toggle instances with: launch_cpu=true / launch_gpu=true
# ─────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  tags = {
    Project     = var.project_name
    ManagedBy   = "terraform"
    Environment = "research"
  }
  bucket_name = var.s3_bucket_name != "" ? var.s3_bucket_name : "${var.project_name}-${random_id.bucket_suffix.hex}"
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# ═══════════════════════════════════════════════════════════════════════════════
# NETWORKING
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = merge(local.tags, { Name = "${var.project_name}-vpc" })
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = merge(local.tags, { Name = "${var.project_name}-igw" })
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}d"
  map_public_ip_on_launch = true
  tags                    = merge(local.tags, { Name = "${var.project_name}-subnet" })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
  tags = merge(local.tags, { Name = "${var.project_name}-rt" })
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "instance" {
  name_prefix = "${var.project_name}-"
  vpc_id      = aws_vpc.main.id
  description = "EDA-KG instance security group"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, { Name = "${var.project_name}-sg" })
}

# ═══════════════════════════════════════════════════════════════════════════════
# S3 BUCKET — data handoff between instances + checkpoint storage
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_s3_bucket" "data" {
  bucket        = local.bucket_name
  force_destroy = true
  tags          = merge(local.tags, { Name = "${var.project_name}-data" })
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    id     = "cleanup-old-artifacts"
    status = "Enabled"
    expiration {
      days = 30
    }
    filter {
      prefix = "runs/"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# (ECR removed — using direct install on DLAMI instead of Docker images)

# ═══════════════════════════════════════════════════════════════════════════════
# IAM — Instance Profile with S3 access (ECR no longer needed)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_iam_role" "instance" {
  name_prefix = "${var.project_name}-"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy" "s3_access" {
  name_prefix = "s3-"
  role        = aws_iam_role.instance.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"]
        Resource = [
          aws_s3_bucket.data.arn,
          "${aws_s3_bucket.data.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "instance" {
  name_prefix = "${var.project_name}-"
  role        = aws_iam_role.instance.name
}

# ═══════════════════════════════════════════════════════════════════════════════
# AMI LOOKUP — Deep Learning AMI for both instances (Ubuntu 22.04)
# CPU instance: uses Python/pip from DLAMI, ignores GPU
# GPU instance: uses full DLAMI stack (CUDA + PyTorch + GPU)
# ═══════════════════════════════════════════════════════════════════════════════

data "aws_ami" "deep_learning" {
  count       = var.ami_id == "" ? 1 : 0
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

locals {
  ami_id = var.ami_id != "" ? var.ami_id : data.aws_ami.deep_learning[0].id
}

# ═══════════════════════════════════════════════════════════════════════════════
# LAUNCH TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_launch_template" "cpu" {
  name_prefix   = "${var.project_name}-cpu-"
  image_id      = local.ami_id
  key_name      = var.key_pair_name

  iam_instance_profile {
    arn = aws_iam_instance_profile.instance.arn
  }

  vpc_security_group_ids = [aws_security_group.instance.id]

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = var.cpu_ebs_size_gb
      volume_type           = "gp3"
      throughput            = 250
      iops                  = 3000
      delete_on_termination = true
    }
  }

  user_data = base64encode(templatefile("${path.module}/../scripts/bootstrap_cpu.sh", {
    aws_region = var.aws_region
    s3_bucket  = aws_s3_bucket.data.bucket
  }))

  tag_specifications {
    resource_type = "instance"
    tags          = merge(local.tags, { Name = "${var.project_name}-cpu", Stage = "1-2" })
  }

  tags = local.tags
}

resource "aws_launch_template" "gpu" {
  name_prefix   = "${var.project_name}-gpu-"
  image_id      = local.ami_id
  key_name      = var.key_pair_name

  iam_instance_profile {
    arn = aws_iam_instance_profile.instance.arn
  }

  vpc_security_group_ids = [aws_security_group.instance.id]

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = var.gpu_ebs_size_gb
      volume_type           = "gp3"
      throughput            = 250
      iops                  = 3000
      delete_on_termination = true
    }
  }

  user_data = base64encode(templatefile("${path.module}/../scripts/bootstrap_gpu.sh", {
    aws_region = var.aws_region
    s3_bucket  = aws_s3_bucket.data.bucket
  }))

  tag_specifications {
    resource_type = "instance"
    tags          = merge(local.tags, { Name = "${var.project_name}-gpu", Stage = "3-5" })
  }

  tags = local.tags
}

# ═══════════════════════════════════════════════════════════════════════════════
# SPOT INSTANCES — toggled via launch_cpu / launch_gpu variables
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_spot_instance_request" "cpu" {
  count                = var.launch_cpu ? 1 : 0
  ami                  = local.ami_id
  spot_price           = var.spot_max_price_cpu
  instance_type        = var.cpu_instance_types[0]
  wait_for_fulfillment = true
  spot_type            = "one-time"
  key_name             = var.key_pair_name

  launch_template {
    id      = aws_launch_template.cpu.id
    version = "$Latest"
  }

  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.instance.id]
  iam_instance_profile   = aws_iam_instance_profile.instance.name

  tags = merge(local.tags, { Name = "${var.project_name}-cpu-spot" })
}

resource "aws_spot_instance_request" "gpu" {
  count                = var.launch_gpu ? 1 : 0
  ami                  = local.ami_id
  spot_price           = var.spot_max_price_gpu
  instance_type        = var.gpu_instance_types[0]
  wait_for_fulfillment = true
  spot_type            = "one-time"

  launch_template {
    id      = aws_launch_template.gpu.id
    version = "$Latest"
  }

  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.instance.id]
  iam_instance_profile   = aws_iam_instance_profile.instance.name

  tags = merge(local.tags, { Name = "${var.project_name}-gpu-spot" })
}
