# Brain-Computer Interfaces

Consumer-grade EEG devices that let you connect your brain to software.

## Neurosity Crown

The device featured in [Fireship's video](https://www.youtube.com/watch?v=GRFXGY4NzxU) "I literally connected my brain to GPT-4 with JavaScript".

### What It Does

- 8-channel EEG headset (highest channel count among consumer BCIs)
- Detects mental states: focus, calm, and custom "kinesis" commands
- Integrates natively with Claude and ChatGPT
- Open JavaScript/Python SDK for developers

### Pricing

| Option | Cost | Notes |
|--------|------|-------|
| Purchase | $1,199 | Increased from $899 in Sept 2024 |
| **Rental** | $99/month | Can switch to purchase later |
| App subscription | $29.99/month or $299.99/year | First year included with purchase |
| SDK/API | Free | Console, API, and SDK always free |

### The Fireship Demo

In the video, Fireship:
1. Trained the Crown to recognize mental patterns ("calm", "focus", custom ones like "bite a lemon")
2. Used feature extraction to isolate brainwave samples for each pattern
3. Triggered OpenAI API calls based on detected mental states
4. e.g., "Bite a Lemon" brain state → sends "tell me a joke" to GPT-4

### Developer Access

- JavaScript SDK (Web and React Native)
- Python SDK
- Kinesis platform for training custom commands
- 4-week developer program available

## Alternatives

### Omi ($89)

Silicon Valley startup Based Hardware's offering announced at CES 2025:
- Worn as necklace or attached to head with medical tape
- Developer version: $70
- Consumer version: $89 (shipping Q2 2025)
- More affordable entry point than Neurosity

## Resources

- [Neurosity](https://neurosity.co/) - Official site
- [Neurosity GitHub](https://github.com/neurosity) - SDKs and examples
- [Fireship Video](https://www.youtube.com/watch?v=GRFXGY4NzxU) - Brain to GPT-4 demo
