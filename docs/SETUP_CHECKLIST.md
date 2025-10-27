# Setup Checklist

Use this checklist to get your HooYouKnow pipeline up and running.

## Pre-requisites

- [ ] Python 3.9 or higher installed
- [ ] Git installed
- [ ] Text editor or IDE (VSCode, PyCharm, etc.)
- [ ] OpenAI API account ([sign up](https://platform.openai.com/signup))
- [ ] Anthropic API account (optional, for Claude) ([sign up](https://console.anthropic.com/))

## Initial Setup

### 1. Project Setup
- [ ] Navigate to the project directory
- [ ] Review README.md to understand the project
- [ ] Check that all files are present (see directory structure below)

### 2. Environment Setup
- [ ] Create Python virtual environment:
  ```bash
  python -m venv venv
  ```
- [ ] Activate virtual environment:
  - Mac/Linux: `source venv/bin/activate`
  - Windows: `venv\Scripts\activate`
- [ ] Verify Python version:
  ```bash
  python --version  # Should be 3.9+
  ```

### 3. Install Dependencies
- [ ] Install required packages:
  ```bash
  pip install -r requirements.txt
  ```
- [ ] Verify installation:
  ```bash
  pip list | grep openai
  pip list | grep anthropic
  ```
- [ ] Install optional dev dependencies (if contributing):
  ```bash
  pip install -e ".[dev]"
  ```

### 4. API Keys Configuration
- [ ] Get OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
- [ ] Get Anthropic API key from [Anthropic Console](https://console.anthropic.com/) (optional)
- [ ] Copy environment template:
  ```bash
  cp config/.env.example config/.env
  ```
- [ ] Edit `config/.env` and add your API keys:
  ```env
  OPENAI_API_KEY=sk-...
  ANTHROPIC_API_KEY=sk-ant-...
  LLM_PROVIDER=claude  # or "openai"
  ```
- [ ] Verify `.env` file is in `.gitignore` (it should be!)

### 5. Configuration Review
- [ ] Review `config/pipeline_config.yaml`
- [ ] Check section definitions match your needs
- [ ] Verify newsletter template is correct
- [ ] Adjust quality thresholds if needed
- [ ] Review model parameters (temperature, max_tokens)

### 6. Test Installation
- [ ] Run configuration validation:
  ```bash
  python -m hooyouknow.cli validate-config
  ```
- [ ] Verify all checks pass ✅
- [ ] Fix any reported issues

## First Run

### 7. Prepare Test Audio
- [ ] Place a test interview audio file in `examples/`
- [ ] Supported formats: .m4a, .mp3, .wav, .mp4, .flac
- [ ] Recommended: Start with a 10-20 minute interview

### 8. Run First Processing
- [ ] Process your test interview:
  ```bash
  python -m hooyouknow.cli process examples/test_interview.m4a \
    --guest-name "Test Guest" \
    --guest-title "Test Title" \
    --guest-company "Test Company"
  ```
- [ ] Check for errors in output
- [ ] Verify files created in `outputs/` directory:
  - [ ] Newsletter markdown file
  - [ ] Transcript file (if save-intermediate is true)
  - [ ] Metadata JSON file

### 9. Review Output Quality
- [ ] Open generated newsletter markdown file
- [ ] Verify sections are properly extracted:
  - [ ] COLD OPEN present
  - [ ] TURNING POINT present
  - [ ] STEAL THIS present
  - [ ] INDUSTRY INSIDER present
  - [ ] IF I WERE YOU present
- [ ] Check content quality and accuracy
- [ ] Review any quality warnings in metadata

### 10. Iterate on Configuration
If output quality needs improvement:
- [ ] Review extracted sections vs. original audio
- [ ] Adjust extraction prompts in `pipeline_config.yaml`
- [ ] Modify temperature/max_tokens if needed
- [ ] Re-run on same audio to test improvements
- [ ] Document any changes made

## Production Setup

### 11. Batch Processing Setup
- [ ] Create `examples/guest_info.json` with your guests
- [ ] Format:
  ```json
  [
    {
      "name": "Guest Name",
      "title": "Job Title",
      "company": "Company Name"
    }
  ]
  ```
- [ ] Test batch processing:
  ```bash
  python -m hooyouknow.cli batch examples/ \
    --guest-info-file examples/guest_info.json
  ```

### 12. Integration Setup (Optional)
- [ ] Set up file watching for automatic processing
- [ ] Configure output directory for your workflow
- [ ] Set up any post-processing scripts
- [ ] Configure backup/archival of processed files

### 13. Git Repository Setup
- [ ] Initialize git repository (if not already):
  ```bash
  git init
  ```
- [ ] Verify `.gitignore` excludes sensitive files:
  - [ ] `config/.env`
  - [ ] `outputs/`
  - [ ] `*.m4a` (audio files)
- [ ] Create initial commit:
  ```bash
  git add .
  git commit -m "Initial commit: HooYouKnow pipeline"
  ```
- [ ] Add remote repository:
  ```bash
  git remote add origin https://github.com/yourusername/hooyouknow-pipeline.git
  ```
- [ ] Push to GitHub:
  ```bash
  git push -u origin main
  ```

### 14. Documentation
- [ ] Update README.md with your specific setup
- [ ] Document any custom configurations
- [ ] Add examples of your processed newsletters
- [ ] Create team documentation if needed

## Testing & Quality Assurance

### 15. Testing
- [ ] Run unit tests:
  ```bash
  pytest tests/
  ```
- [ ] Run integration tests (if available)
- [ ] Test error handling:
  - [ ] Try invalid audio file
  - [ ] Try very short audio
  - [ ] Try missing guest information
- [ ] Verify all edge cases work as expected

### 16. Performance Testing
- [ ] Process multiple interviews to check consistency
- [ ] Note processing times for different audio lengths
- [ ] Monitor API costs for your usage pattern
- [ ] Verify no memory leaks in batch processing

## Maintenance

### 17. Regular Maintenance Tasks
- [ ] Review and update extraction prompts monthly
- [ ] Check API usage and costs
- [ ] Update dependencies regularly:
  ```bash
  pip install --upgrade -r requirements.txt
  ```
- [ ] Review and refine quality thresholds
- [ ] Archive old outputs periodically

### 18. Monitoring Setup (Optional)
- [ ] Set up logging monitoring
- [ ] Configure alerts for failures
- [ ] Track success/failure rates
- [ ] Monitor processing costs

## Advanced Features

### 19. Custom Sections (Optional)
- [ ] Define new section in `pipeline_config.yaml`
- [ ] Test extraction with new section
- [ ] Update newsletter template if needed

### 20. Custom Output Formats (Optional)
- [ ] Implement custom formatter in `formatting.py`
- [ ] Add to output configuration
- [ ] Test new format

### 21. Workflow Automation (Optional)
- [ ] Set up automated file processing
- [ ] Configure scheduled batch runs
- [ ] Integrate with cloud storage
- [ ] Set up CI/CD pipeline

## Troubleshooting Checklist

If something doesn't work:
- [ ] Check Python version (must be 3.9+)
- [ ] Verify all dependencies installed
- [ ] Confirm API keys are correct
- [ ] Check `.env` file location (in `config/` directory)
- [ ] Review logs for error messages
- [ ] Try with a different audio file
- [ ] Check internet connectivity
- [ ] Verify API quotas/limits not exceeded
- [ ] Review GitHub issues for known problems

## Success Criteria

Your setup is complete when:
- [ ] Configuration validation passes
- [ ] Test interview processes successfully
- [ ] Output files are created correctly
- [ ] Extracted sections are accurate and complete
- [ ] No errors in processing
- [ ] Costs are as expected (~$0.23 per 25-min interview)
- [ ] You can process interviews consistently

## Next Steps After Setup

1. **Process Your Real Interviews**
   - Start with a few interviews
   - Review and refine prompts
   - Build confidence in the system

2. **Optimize Your Workflow**
   - Set up batch processing
   - Automate routine tasks
   - Integrate with your existing tools

3. **Share with Team**
   - Document your customizations
   - Train team members
   - Establish quality review process

4. **Monitor and Improve**
   - Track quality metrics
   - Gather feedback
   - Continuously refine prompts

## Support Resources

- **Documentation**: README.md, QUICKSTART.md, ARCHITECTURE.md
- **Examples**: examples/ directory
- **Issues**: Create GitHub issue for bugs
- **Community**: Contribute improvements back

---

## Quick Reference Commands

```bash
# Validate setup
python -m hooyouknow.cli validate-config

# Process single interview
hooyouknow process audio.m4a -n "Name" -t "Title"

# Batch process
hooyouknow batch /path/to/audio/ --guest-info-file guests.json

# Run tests
pytest tests/

# Update dependencies
pip install --upgrade -r requirements.txt
```

---

🎉 **Congratulations!** You're ready to transform interviews into newsletters!
