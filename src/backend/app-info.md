# Message Analyzer

## Overview
The Message Analyzer is a tool designed to analyze conversations and detect specific patterns of interaction between speakers. Using natural language processing, it identifies key behaviors and content including age mentions, meetup requests, gift-giving, and media sharing.

## Features
- Processes CSV files containing conversations with timestamp, speaker, and message columns
- Analyzes five key aspects of conversations:
  1. Age Information: Detects when participants share their age
  2. Age Inquiries: Identifies when participants ask about each other's age
  3. Meetup Requests: Flags discussions about meeting in person
  4. Gift Exchanges: Identifies mentions of gifts or purchases for others
  5. Media Sharing: Detects mentions of photos, videos, or other media content

## Input Format
- File Type: CSV
- Required Columns:
  - Timestamp: Time of message (format: YYYY-MM-DD HH:MM)
  - Speaker: Name of the person sending the message
  - Message: Content of the message

## Output
The analysis results are presented in two sections:
1. Summary Table
   - Questions analyzed with YES/NO answers
   - Evidence quotes for positive matches
   - Color-coded indicators for different types of interactions
2. Full Conversation View
   - Complete conversation with timestamps
   - Visual indicators showing where evidence was found

## Use Cases
- Content moderation
- Conversation pattern analysis
- Behavioral monitoring
- Safety and security checks

## Notes
- The analyzer processes one conversation at a time
- Results are displayed immediately after processing
- Evidence is provided with specific message quotes