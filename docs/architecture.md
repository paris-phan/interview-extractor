# System Architecture & Design

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
├─────────────────────────────────────────────────────────────┤
│  CLI (cli.py)          │  Python API (pipeline.py)          │
│  Command-line tool     │  Programmatic interface            │
└────────────┬───────────┴────────────┬────────────────────────┘
             │                        │
             └────────────┬───────────┘
                          │
            ┌─────────────▼──────────────┐
            │   Pipeline Orchestrator    │
            │      (pipeline.py)         │
            │  - Coordinates workflow    │
            │  - Manages state           │
            │  - Tracks costs/metrics    │
            └─────────────┬──────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼────────┐ ┌─────▼──────┐ ┌───────▼────────┐
│ Transcription  │ │ Extraction │ │  Formatting    │
│   Module       │ │   Module   │ │    Module      │
│ (transcription │ │ (extraction│ │ (formatting.py)│
│     .py)       │ │     .py)   │ │                │
│                │ │            │ │                │
│ - Audio → Text │ │ - Text →   │ │ - Sections →   │
│ - Whisper API  │ │   Sections │ │   Newsletter   │
│ - Timestamps   │ │ - Claude/  │ │ - MD/HTML/JSON │
│                │ │   GPT-4    │ │                │
└────────┬───────┘ └─────┬──────┘ └────────┬───────┘
         │               │                 │
         └───────────────┼─────────────────┘
                         │
            ┌────────────▼────────────┐
            │    Utilities & Config   │
            │       (utils.py)        │
            │  - Configuration        │
            │  - Logging              │
            │  - Cost calculation     │
            │  - File management      │
            └─────────────────────────┘
```

## Data Flow

```
                    INPUT
                      │
                Audio File
              (.m4a, .mp3, etc.)
                      │
                      ▼
        ┌─────────────────────────┐
        │  1. TRANSCRIPTION       │
        │  ─────────────────      │
        │  • Load audio file      │
        │  • Send to Whisper API  │
        │  • Receive transcript   │
        │  • Extract segments     │
        │  • Add timestamps       │
        └────────┬────────────────┘
                 │
           Transcript Text
        (with optional segments)
                 │
                 ▼
        ┌─────────────────────────┐
        │  2. EXTRACTION          │
        │  ──────────────         │
        │  • Build prompt         │
        │  • Send to LLM          │
        │  • Parse JSON response  │
        │  • Validate sections    │
        │  • Check quality        │
        └────────┬────────────────┘
                 │
          Structured Sections
         {COLD_OPEN: "...",
          TURNING_POINT: "...",
          ...}
                 │
                 ▼
        ┌─────────────────────────┐
        │  3. FORMATTING          │
        │  ──────────────         │
        │  • Select format        │
        │  • Apply template       │
        │  • Generate content     │
        │  • Save to file         │
        └────────┬────────────────┘
                 │
                 ▼
                OUTPUT
                 │
    ┌────────────┼────────────┐
    │            │            │
Newsletter.md  Metadata   Transcript
               .json      .txt
```

## Component Details

### 1. Transcription Module (`transcription.py`)

**Responsibilities:**
- Audio file handling and validation
- API communication with OpenAI Whisper
- Retry logic for failed requests
- Timestamp extraction
- Transcript saving

**Key Classes:**
- `AudioTranscriber`: Main transcription handler

**Key Methods:**
- `transcribe()`: Single transcription attempt
- `transcribe_with_retry()`: Transcription with error handling
- `save_transcript()`: Save transcript to file

**Dependencies:**
- OpenAI Python SDK
- pydub (optional, for audio duration)

### 2. Extraction Module (`extraction.py`)

**Responsibilities:**
- Prompt engineering and construction
- LLM API communication (Claude or GPT-4)
- Response parsing and validation
- Section quality checking

**Key Classes:**
- `ContentExtractor`: Main extraction handler

**Key Methods:**
- `extract_sections()`: Extract all sections from transcript
- `_build_extraction_prompt()`: Construct LLM prompt
- `_call_claude()` / `_call_openai()`: Provider-specific API calls
- `validate_sections()`: Quality validation

**Dependencies:**
- Anthropic Python SDK (for Claude)
- OpenAI Python SDK (for GPT-4)

### 3. Formatting Module (`formatting.py`)

**Responsibilities:**
- Newsletter template management
- Content formatting for different outputs
- File generation and saving

**Key Classes:**
- `NewsletterFormatter`: Main formatting handler

**Key Methods:**
- `format_newsletter()`: Format sections into newsletter
- `_format_markdown()`: Markdown-specific formatting
- `_format_html()`: HTML conversion
- `_format_json()`: JSON serialization
- `save_newsletter()`: Write formatted content to file

### 4. Pipeline Orchestrator (`pipeline.py`)

**Responsibilities:**
- Component initialization
- Workflow coordination
- State management
- Statistics tracking
- Error handling
- Batch processing

**Key Classes:**
- `InterviewPipeline`: Main pipeline controller

**Key Methods:**
- `process_interview()`: Single interview processing
- `batch_process()`: Multiple interview processing
- `_initialize_components()`: Setup all modules

### 5. Utilities (`utils.py`)

**Responsibilities:**
- Configuration loading
- Environment variable management
- Logging setup
- Cost calculation
- File operations
- Helper functions

**Key Functions:**
- `load_config()`: Load YAML configuration
- `load_env_vars()`: Load environment variables
- `calculate_cost()`: Estimate processing costs
- `setup_logging()`: Configure logging

**Key Classes:**
- `ProcessingStats`: Track processing metrics

## Configuration System

```
config/
├── .env                      # Environment variables (gitignored)
│   ├── OPENAI_API_KEY
│   ├── ANTHROPIC_API_KEY
│   ├── LLM_PROVIDER
│   └── ...
│
└── pipeline_config.yaml      # Pipeline configuration
    ├── sections[]            # Section definitions
    │   ├── name
    │   ├── title
    │   ├── question_pattern
    │   ├── extraction_prompt
    │   └── optional
    ├── newsletter{}          # Newsletter metadata
    ├── extraction{}          # Extraction settings
    ├── transcription{}       # Transcription settings
    ├── output{}             # Output configuration
    └── quality_check{}       # Quality validation rules
```

## Error Handling Strategy

```
┌─────────────────────────────────────┐
│         Error Occurs                │
└────────────┬────────────────────────┘
             │
             ▼
    ┌────────────────────┐
    │  Retry Strategy?   │
    └────┬───────┬───────┘
         │       │
    Yes  │       │  No
         │       │
         ▼       ▼
    ┌─────┐  ┌──────────┐
    │Retry│  │  Log     │
    │ up  │  │  Error   │
    │ to  │  │  Return  │
    │ 3x  │  │  Failure │
    └─────┘  └──────────┘
         │
         ▼
    ┌────────────┐
    │  Success?  │
    └─┬────────┬─┘
      │        │
   Yes│        │No
      │        │
      ▼        ▼
   Continue  Fail &
   Pipeline  Report
```

## Quality Control Flow

```
Sections Extracted
       │
       ▼
┌──────────────────┐
│  Validate Each   │
│     Section      │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Required   Optional
Section    Section
    │         │
    ▼         ▼
Present?   Present?
    │         │
No  │  Yes    │  Any
    ▼         ▼
  Error    Continue
    │
    └──────┬─────────┐
           │         │
           ▼         ▼
      Word Count  Content
      In Range?   Quality
           │         │
      ┌────┴────┐    │
   No │         │Yes │
      ▼         ▼    ▼
   Warning   Pass  Review
```

## Deployment Options

### Option 1: Local Execution
```
User Machine
  │
  ├── Python Environment
  ├── Source Code
  ├── Configuration
  └── API Keys
```

### Option 2: Server/Cloud
```
Cloud Server (AWS/GCP/Azure)
  │
  ├── Docker Container
  │   ├── Python Runtime
  │   ├── Application Code
  │   └── Dependencies
  │
  ├── Environment Variables
  │   └── API Keys (secure)
  │
  └── Storage
      ├── Input: Audio files
      └── Output: Newsletters
```

### Option 3: CI/CD Pipeline
```
Git Repository
  │
  ▼
GitHub Actions / CI/CD
  │
  ├── Trigger on commit/schedule
  ├── Run pipeline on new audio
  ├── Validate output
  └── Deploy to production
```

## Performance Characteristics

**Transcription:**
- Speed: ~0.1x real-time (10 min audio = 1 min processing)
- Bottleneck: API request time
- Optimization: Parallel processing for batch

**Extraction:**
- Speed: 10-30 seconds per interview
- Bottleneck: LLM inference time
- Optimization: Prompt efficiency, token reduction

**Formatting:**
- Speed: < 1 second
- Bottleneck: None (CPU-bound, very fast)

**Total Pipeline:**
- 20-minute interview: ~3-5 minutes total processing
- Dominated by API call latency

## Scalability Considerations

**Current Design:**
- Sequential processing
- Single-threaded
- API rate limits respected

**Scaling Strategies:**
1. **Parallel Processing**: Process multiple interviews simultaneously
2. **Caching**: Cache transcripts to avoid re-transcription
3. **Batch API Calls**: Group requests where possible
4. **Queue System**: Implement job queue for large batches
5. **Distributed**: Multiple workers for high-volume processing

## Security Model

```
┌─────────────────────────────────────┐
│          Security Layers            │
├─────────────────────────────────────┤
│  1. API Key Management              │
│     - Stored in .env (gitignored)   │
│     - Never hardcoded               │
│     - Environment variable access   │
├─────────────────────────────────────┤
│  2. Input Validation                │
│     - File type checking            │
│     - Path sanitization             │
│     - Size limits                   │
├─────────────────────────────────────┤
│  3. API Communication               │
│     - HTTPS only                    │
│     - Timeout handling              │
│     - Rate limiting                 │
├─────────────────────────────────────┤
│  4. Output Security                 │
│     - Safe file naming              │
│     - Directory restrictions        │
│     - Permission management         │
└─────────────────────────────────────┘
```

## Extension Points

The architecture is designed for easy extension:

1. **New Transcription Providers**: Add to `transcription.py`
2. **New LLM Providers**: Add to `extraction.py`
3. **New Output Formats**: Add to `formatting.py`
4. **New Sections**: Configure in YAML (no code change)
5. **Custom Validation**: Add to quality_check in config
6. **Preprocessing Hooks**: Add before transcription
7. **Postprocessing Hooks**: Add after formatting

This modular design ensures the system can evolve with your needs!
