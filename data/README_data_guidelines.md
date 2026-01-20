# Sinhala Letters Dataset Guidelines (v2 Schema)

This document describes the schema and guidelines for creating and maintaining the Sinhala letter dataset used for retrieval-augmented generation.

## Dataset Schema (v2)

The dataset CSV (`sinhala_letters_v2.csv`) uses the following columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `id` | string | Yes | Unique identifier (e.g., "REQ001", "STRUCT001") |
| `letter_category` | string | Yes | Category of letter (see categories below) |
| `doc_type` | string | Yes | Type of document: `example`, `structure`, `section_template` |
| `register` | string | Yes | Formality level: `formal`, `very_formal` |
| `language` | string | Yes | Language code: `si` (Sinhala) |
| `source` | string | Yes | Origin: `synthetic`, `curated`, `user_generated` |
| `title` | string | Yes | Short title/subject in Sinhala |
| `content` | string | Yes | Full text content |
| `tags` | string | No | Comma-separated tags for additional filtering |
| `rating` | float | No | Quality rating (1-5) for user-generated content |

## Letter Categories

| Category | Sinhala | Description |
|----------|---------|-------------|
| `request` | ඉල්ලීම | Request letters (leave, transfer, resources) |
| `apology` | ක්ෂමාව | Apology letters |
| `invitation` | ආරාධනා | Invitation letters |
| `complaint` | පැමිණිලි | Complaint letters |
| `application` | අයදුම්පත් | Job/program applications |
| `general` | සාමාන්‍ය | General formal letters |
| `notification` | දැනුම්දීම | Notification/announcement letters |
| `appreciation` | ස්තුති | Appreciation/thank you letters |

## Document Types

### 1. `example` - Complete Letter Examples
Full letter examples that demonstrate the complete structure and content.

```
- Should be complete, ready-to-use letters
- Include all standard sections (salutation, body, closing, signature)
- Can be synthetic (created for training) or curated (from real sources)
- Used to show the LLM the expected output format
```

### 2. `structure` - Letter Structure Templates
Templates showing the structural skeleton of a letter type.

```
- Shows section ordering and placement
- Uses placeholders like [ලබන්නාගේ නම], [දිනය], etc.
- Focuses on format rather than specific content
- Helps ensure correct formal structure
```

### 3. `section_template` - Reusable Sections
Small, reusable sections that can be mixed and matched.

```
- Individual components: greetings, closings, standard phrases
- Category-specific standard paragraphs
- Formal phrases and expressions
- Useful for maintaining consistency across letters
```

## Register Levels

### `formal`
Standard formal language appropriate for most official correspondence.
- Used for: Office communications, general requests, applications
- Tone: Respectful, professional

### `very_formal`
Highly formal language for government, legal, or ceremonial contexts.
- Used for: Government letters, legal documents, formal ceremonies
- Tone: Deferential, traditional, elaborate

## Creating New Entries

### Guidelines for Examples

1. **Completeness**: Include all sections of a formal letter
2. **Authenticity**: Use realistic but anonymized names/details
3. **Variety**: Create examples for different sub-scenarios within each category
4. **Quality**: Ensure proper Sinhala grammar and formal register

### Guidelines for Structure Templates

1. **Clarity**: Use clear placeholder markers
2. **Flexibility**: Make templates adaptable to various scenarios
3. **Completeness**: Include all required structural elements
4. **Instructions**: Add comments for optional sections

### Standard Placeholder Format

Use square brackets with Sinhala labels:
- `[ලබන්නාගේ නම]` - Recipient's name
- `[ලබන්නාගේ තනතුර]` - Recipient's position
- `[යවන්නාගේ නම]` - Sender's name
- `[දිනය]` - Date
- `[ලිපිනය]` - Address
- `[මාතෘකාව]` - Subject
- `[අන්තර්ගතය]` - Content/body
- `[හේතුව]` - Reason/purpose

## Sample Structure Template

```
[දිනය]

[ලබන්නාගේ නම]
[ලබන්නාගේ තනතුර]
[ආයතනයේ නම]
[ලිපිනය]

ගරු මහත්මයාණෙනි/මහත්මිය,

මාතෘකාව: [මාතෘකාව]

[ආමන්ත්‍රණ වාක්‍යය]

[අන්තර්ගතය - ඡේද 1]

[අන්තර්ගතය - ඡේද 2]

[අවසාන ඉල්ලීම/ස්තුතිය]

ගෞරවයෙන්,

[යවන්නාගේ නම]
[තනතුර]
[සම්බන්ධතා විස්තර]
```

## Tags Usage

Tags provide additional filtering capabilities. Use comma-separated values:

```
"leave,annual,employee" - For annual leave request
"complaint,service,delay" - For service delay complaint
"job,application,graduate" - For graduate job application
```

### Recommended Tag Categories
- **Topic**: leave, transfer, promotion, complaint, appreciation
- **Context**: employee, student, customer, citizen, official
- **Urgency**: urgent, routine, followup
- **Sector**: government, private, education, healthcare

## File Organization

```
data/
├── sinhala_letters_v2.csv      # Main dataset (v2 schema)
├── README_data_guidelines.md   # This file
├── examples/                   # Additional example files
│   ├── request_examples.md
│   ├── complaint_examples.md
│   └── ...
└── templates/                  # Structure templates
    ├── formal_structures.md
    └── very_formal_structures.md
```

## Quality Checklist

Before adding new entries, verify:

- [ ] Proper Sinhala spelling and grammar
- [ ] Appropriate register (formal/very_formal)
- [ ] Correct category assignment
- [ ] Complete content (no missing sections)
- [ ] Unique ID that follows naming convention
- [ ] Relevant tags assigned
- [ ] No sensitive personal information

## ID Naming Convention

Format: `{CATEGORY_PREFIX}{NUMBER}`

| Category | Prefix | Example |
|----------|--------|---------|
| Request | REQ | REQ001 |
| Apology | APO | APO001 |
| Invitation | INV | INV001 |
| Complaint | CMP | CMP001 |
| Application | APP | APP001 |
| General | GEN | GEN001 |
| Notification | NOT | NOT001 |
| Appreciation | APR | APR001 |
| Structure | STRUCT | STRUCT001 |
| Section | SEC | SEC001 |
