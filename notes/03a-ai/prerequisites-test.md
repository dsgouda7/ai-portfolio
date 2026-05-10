# Prerequisites Self-Assessment for LLM Fundamentals Track

> **Purpose**: This test helps you determine if you're ready to start the LLM Fundamentals track. If you can answer most of these questions, you have the foundation needed to succeed. If not, we recommend reviewing the linked resources first.

---

## Part 1: Python Programming (Required)

### Question 1: Basic Syntax
Can you read and understand this Python code?
```python
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers) if len(numbers) > 0 else 0

scores = [85, 92, 78, 95]
print(f"Average score: {calculate_average(scores)}")
```

**What it tests**: Functions, list operations, conditional expressions, f-strings

**If you're unsure**: Review Python basics on [python.org/about/gettingstarted](https://www.python.org/about/gettingstarted/)

---

### Question 2: Data Structures
Can you explain what this code does?
```python
embeddings = {
    "cat": [0.2, 0.8, 0.1],
    "dog": [0.3, 0.7, 0.2],
    "fish": [-0.5, 0.1, 0.9]
}

for word, vector in embeddings.items():
    print(f"{word}: {len(vector)} dimensions")
```

**What it tests**: Dictionaries, lists, iteration, string formatting

**If you're unsure**: You'll work with dictionaries and lists extensively in this track. Review Python data structures before continuing.

---

### Question 3: Working with APIs
Can you understand this API call structure?
```python
import requests

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
result = response.json()
```

**What it tests**: HTTP POST requests, headers, JSON payloads

**If you're unsure**: Every chapter uses OpenAI/Anthropic APIs. You should understand HTTP requests and JSON before starting.

---

## Part 2: Very Basic ML Concepts (Will Be Taught Here)

> **Good news**: You DON'T need deep ML knowledge to start this track. The questions below test if you've heard these terms before. If you haven't, **that's okay** — Chapter 1's Appendix A defines all of them.

### Question 4: Have You Heard These Terms?
Mark which terms sound familiar (no need to explain them in depth):

- [ ] Training (teaching a model by showing it examples)
- [ ] Parameters (numbers in a model that get adjusted)
- [ ] Neural network (a model inspired by how brains work)
- [ ] Prediction (model's output or guess)

**If you checked 0-1**: You're brand new to ML — welcome! Start with Chapter 1 and read Appendix A carefully.

**If you checked 2-3**: You have a basic sense of ML — perfect starting point.

**If you checked all 4**: You're well-prepared for this track.

---

### Question 5: The Concept of "Learning from Data"
Which statement best describes how you think ML models work?

A) They memorize a database of facts and look them up
B) They find patterns in examples and apply those patterns to new cases
C) They are programmed with explicit rules by engineers
D) They randomly guess until they get the right answer

**Answer**: B — Models learn patterns from training data, then apply those patterns to new inputs.

**If you answered A, C, or D**: That's okay! Chapter 1 explains how models actually work. The key insight: models learn **patterns**, not **rules** or **lookups**.

---

## Part 3: HTTP & APIs (Required)

### Question 6: Understanding API Responses
If an API returns this JSON:
```json
{
  "choices": [
    {"message": {"content": "The capital of France is Paris."}}
  ],
  "usage": {"total_tokens": 25}
}
```

How would you extract the text "The capital of France is Paris." in Python?

**Possible answer**: `response["choices"][0]["message"]["content"]`

**What it tests**: JSON navigation, bracket notation

**If you're unsure**: You'll parse API responses in every chapter. Review JSON basics.

---

## Scoring Guide

### ✅ Ready to Start
- Comfortable with Python (Questions 1-3)
- Familiar with HTTP/JSON (Question 6)
- Heard 2+ ML terms from Question 4 (or willing to learn)

**Action**: Jump to [Chapter 1: Transformer Architecture](ch01-transformer-architecture/transformer-architecture.md)

---

### ⚠️ Almost Ready
- Comfortable with Python
- Shaky on APIs/JSON
- Unfamiliar with ML terms

**Action**: Skim a quick HTTP/JSON tutorial (15 min), then start Chapter 1. Read Appendix A carefully.

---

### ❌ Not Quite Ready Yet
- Struggled with Python questions (1-3)
- Unfamiliar with APIs

**Action**: Complete a Python basics course first:
- [python.org Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python: Python Basics](https://realpython.com/learning-paths/python-basics/)
- [Codecademy: Learn Python 3](https://www.codecademy.com/learn/learn-python-3)

Then come back and retake this test.

---

## Next Steps

### If You're Ready:
1. **Set up your environment**: Python 3.9+, `pip install openai anthropic`
2. **Get API keys**: [OpenAI](https://platform.openai.com/api-keys), [Anthropic](https://console.anthropic.com/)
3. **Start Chapter 1**: [Transformer Architecture](ch01-transformer-architecture/transformer-architecture.md)

### If You Need Help:
- **Python refresher**: [Python Cheatsheet](https://www.pythoncheatsheet.org/)
- **HTTP/API basics**: [HTTP Crash Course (MDN)](https://developer.mozilla.org/en-US/docs/Web/HTTP)
- **JSON tutorial**: [Learn JSON in 10 Minutes](https://www.json.org/json-en.html)

---

## FAQ

**Q: I don't know what "embeddings" or "gradients" are. Can I still start?**
A: **Yes!** Those terms are defined in Chapter 1, Appendix A. You'll learn them as you go.

**Q: I've never trained a machine learning model before. Is that a problem?**
A: **No.** This track is about *using* pre-trained LLMs, not training them from scratch. You'll learn *how* they're trained conceptually, but you won't write training code.

**Q: I don't have a CS degree. Will I understand this?**
A: **Yes, if you know Python.** The track is written for practitioners, not researchers. Math is kept minimal and intuition-first.

**Q: Do I need a GPU?**
A: **No.** All examples use APIs (OpenAI, Anthropic, Cohere). You'll make HTTP requests, not run local models.

---

**Ready to begin?** → [Chapter 1: Transformer Architecture](ch01-transformer-architecture/transformer-architecture.md)
