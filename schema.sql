-- FairBench database schema
-- Run this on Neon to create the tables

CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    published BOOLEAN DEFAULT true,
    
    -- Model info
    model_name TEXT NOT NULL,
    quantization TEXT NOT NULL,
    
    -- Hardware info
    cpu_model TEXT,
    cpu_cores INTEGER,
    cpu_threads INTEGER,
    ram_gb FLOAT,
    
    gpu_type TEXT DEFAULT 'none',
    gpu_model TEXT,
    gpu_vram_gb FLOAT,
    
    os TEXT,
    
    -- Quality metrics
    quality_mmlu FLOAT DEFAULT 0,
    quality_hellaswag FLOAT DEFAULT 0,
    quality_gsm8k FLOAT DEFAULT 0,
    
    -- Speed metrics
    speed_ttft_ms FLOAT DEFAULT 0,
    speed_tpot_ms FLOAT DEFAULT 0,
    speed_throughput FLOAT DEFAULT 0,
    
    -- Memory metrics
    memory_vram_peak_gb FLOAT DEFAULT 0,
    memory_cpu_ram_peak_gb FLOAT DEFAULT 0,
    memory_model_file_gb FLOAT DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_reports_published ON reports (published);
CREATE INDEX idx_reports_model ON reports (model_name);
CREATE INDEX idx_reports_gpu ON reports (gpu_model);
CREATE INDEX idx_reports_timestamp ON reports (timestamp DESC);
CREATE INDEX idx_reports_quality ON reports (quality_mmlu DESC);

-- View for leaderboard (top reports by quality)
CREATE OR REPLACE VIEW leaderboard AS
SELECT 
    id, timestamp, model_name, quantization,
    cpu_model, cpu_cores, ram_gb,
    gpu_type, gpu_model, gpu_vram_gb,
    quality_mmlu, quality_hellaswag, quality_gsm8k,
    speed_ttft_ms, speed_tpot_ms, speed_throughput,
    memory_vram_peak_gb, memory_cpu_ram_peak_gb
FROM reports
WHERE published = true
ORDER BY quality_mmlu DESC, speed_throughput DESC;

-- Function to get average stats by model
CREATE OR REPLACE FUNCTION get_model_stats(model_name_param TEXT)
RETURNS TABLE (
    avg_mmlu FLOAT,
    avg_hellaswag FLOAT,
    avg_gsm8k FLOAT,
    avg_ttft_ms FLOAT,
    avg_tpot_ms FLOAT,
    avg_throughput FLOAT,
    avg_vram_gb FLOAT,
    sample_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        AVG(quality_mmlu),
        AVG(quality_hellaswag),
        AVG(quality_gsm8k),
        AVG(speed_ttft_ms),
        AVG(speed_tpot_ms),
        AVG(speed_throughput),
        AVG(memory_vram_peak_gb),
        COUNT(*)
    FROM reports
    WHERE model_name = model_name_param AND published = true;
END;
$$ LANGUAGE plpgsql;
