---
title: Real-Time Intelligent Summarization Platform
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
hardware: t4-small
---

# Real-Time Intelligent Summarization Platform

Production-grade summarization system with hierarchical map-reduce architecture, multi-model ensemble, and hallucination control.

## Features

- Text summarization (unlimited length)
- YouTube video summarization
- Query-focused summaries
- Multiple summary lengths (Short/Medium/Detailed)
- Confidence scoring with source alignment
- Key point extraction
- Source citations
- Result caching

## Technical Implementation

- Hierarchical Map-Reduce for long documents
- Multi-model ensemble (BART, Pegasus, T5)
- TF-IDF relevance scoring for query focus
- Adaptive token-aware chunking
- Topic boundary detection

## Models

- facebook/bart-large-cnn (Primary)
- google/pegasus-cnn_dailymail (Optional)
- t5-large (Optional)

## Usage

1. Select input type (Text or YouTube URL)
2. Paste content or URL
3. Optionally add a focus query
4. Select mode and summary length
5. Click Generate Summary

## Modes

- Student: Simplified explanations
- Researcher: Detailed technical analysis
- Business: Executive summaries
- Beginner: Basic explanations
- Expert: Comprehensive coverage