# Production Culinary Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade every culinary LLM instruction so the chatbot behaves like a production-ready food consultant: grounded, concise, bilingual where practical, natural in tone, and free of internal retrieval/debug terminology.

**Architecture:** Add one shared prompt-policy module with reusable culinary role, grounding, language, and user-facing style rules. Each existing agent keeps a focused task-specific prompt that composes the shared policy, preserving the current workflow and structured-output interfaces while improving behavior through explicit prompt contracts and regression tests.

**Tech Stack:** Python 3.11+, Pydantic, LangChain/OpenAI structured output, LangGraph, pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-production-culinary-prompts-design.md`

## Global Constraints

- Keep restaurant/recipe recommendations grounded in retrieved/project-provided candidates only.
- Never expose embeddings, vector search, ranking scores, fusion scores, modality scores, candidate IDs, agent names, or retrieval internals in normal user-facing explanations.
- Never claim live popularity, trends, opening hours, availability, or similar current facts unless explicit evidence is supplied in the payload.
- Respond in the same language as the user's request when practical.
- Use a professional, approachable culinary-consultant tone.
- Return at most three recommendations and do not pad with weak alternatives when fewer strong matches exist.
- Nutrition reasoning must not invent calories, macros, allergen guarantees, medical benefits, or health outcomes.
- Do not change retrieval architecture, Chroma schemas, embedding models, ranking weights, or dataset formats in this work.

## Review Focus

- Mixed-language query such as Vietnamese with an English cuisine name should still produce a response in the dominant user language; test this in the recommender prompt contract.
- A candidate payload containing internal numeric scores must not cause those score names to appear in the final explanation prompt; test this in the recommender prompt contract.
- A candidate with a rating but no popularity evidence must not be described as trending or popular; test this in the trend prompt contract.
- Nutrition analysis with sparse ingredient metadata must explicitly avoid invented nutrient numbers; test this in the nutrition prompt contract.
- Preference extraction containing wording such as “prefer” versus “must/only/chỉ/bắt buộc” must preserve soft versus strict behavior; test this in the preference prompt contract.

---

### Task 1: Shared Culinary Prompt Policy

**Files:**
- Create: `src/agents/prompt_policy.py`
- Create or modify: `tests/test_prompt_policy.py`

**Interfaces:**
- Consumes: no new runtime dependencies.
- Produces: module-level strings `CULINARY_ROLE_POLICY`, `GROUNDING_POLICY`, `LANGUAGE_POLICY`, `USER_FACING_STYLE_POLICY`, and helper `compose_prompt(task_instruction: str) -> str` for later tasks.

- [ ] **Step 1: Write failing tests for the shared policy**

Add tests that assert the exported policies contain the required production contracts: culinary specialist role, grounding/no fabrication, same-language behavior, professional/approachable tone, and forbidden internal-system terminology.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest tests/test_prompt_policy.py -v`

Expected: FAIL because `src.agents.prompt_policy` does not yet exist.

- [ ] **Step 3: Implement `src/agents/prompt_policy.py`**

Create the four reusable policy strings and `compose_prompt(task_instruction: str) -> str`. The helper should concatenate the shared policies with the task-specific instruction without changing any agent schema or payload shape.

- [ ] **Step 4: Run the focused tests and verify pass**

Run: `pytest tests/test_prompt_policy.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agents/prompt_policy.py tests/test_prompt_policy.py
git commit -m "feat: add shared culinary prompt policy"
```

### Task 2: Preference Extraction Prompt

**Files:**
- Modify: `src/ui/chatbot.py`
- Modify: `tests/test_chatbot.py`

**Interfaces:**
- Consumes: `compose_prompt(task_instruction: str) -> str` from Task 1.
- Produces: improved `PREFERENCE_PROMPT`; `extract_preferences(query: str, llm) -> UserPreferences` signature remains unchanged.

- [ ] **Step 1: Write failing tests for preference prompt behavior**

Add tests that inspect/capture the prompt passed to the fake structured LLM and assert it instructs the model to:
- preserve soft preferences by default;
- use `strict_constraints` only for explicit mandatory wording such as `must`, `only`, `chỉ`, or `bắt buộc`;
- understand Vietnamese and English culinary terminology;
- avoid inferring unstated location, budget, allergies, dietary status, or other constraints;
- use empty lists/null for unspecified fields.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest tests/test_chatbot.py -k "preference_prompt or strict_constraints" -v`

Expected: FAIL because the current prompt is too limited.

- [ ] **Step 3: Update `PREFERENCE_PROMPT` in `src/ui/chatbot.py`**

Compose it from the shared policy plus a focused extraction instruction. Do not change `UserPreferences`, retrieval behavior, or query payload shape.

- [ ] **Step 4: Run the focused tests and verify pass**

Run: `pytest tests/test_chatbot.py -k "preference_prompt or strict_constraints" -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ui/chatbot.py tests/test_chatbot.py
git commit -m "feat: strengthen culinary preference extraction prompt"
```

### Task 3: Profile, Evidence, Style, and Nutrition Agent Prompts

**Files:**
- Modify: `src/agents/profile.py`
- Modify: `src/agents/trends.py`
- Modify: `src/agents/style.py`
- Modify: `src/agents/nutrition.py`
- Create or modify: `tests/test_agent_prompts.py`

**Interfaces:**
- Consumes: `compose_prompt(task_instruction: str) -> str` from Task 1.
- Produces: stronger `PROMPT` constants; existing function signatures and `AnalysisOutput`/`ProfileOutput` schemas remain unchanged.

- [ ] **Step 1: Write failing prompt-contract tests**

Add tests asserting:
- profile prompt prioritizes hard constraints, separates them from soft preferences, and forbids persona assumptions;
- trends prompt reframes work as evidence-fit analysis and forbids unsupported `trending`, `viral`, or live-popularity claims;
- style prompt grounds atmosphere/visual language in supplied descriptions and forbids CLIP/embedding/modality terminology;
- nutrition prompt preserves explicit dietary constraints and forbids invented calories, macros, allergen guarantees, medical benefits, and health outcomes.

Include a regression assertion that a mere rating is not equivalent to popularity evidence.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest tests/test_agent_prompts.py -v`

Expected: FAIL because the existing prompts do not contain these contracts.

- [ ] **Step 3: Update the four agent prompts**

Compose each prompt from the shared policy plus its task-specific instruction. Do not change any agent function signature or structured-output schema.

- [ ] **Step 4: Run the focused tests and verify pass**

Run: `pytest tests/test_agent_prompts.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agents/profile.py src/agents/trends.py src/agents/style.py src/agents/nutrition.py tests/test_agent_prompts.py
git commit -m "feat: harden culinary analysis agent prompts"
```

### Task 4: Production Recommendation Prompt

**Files:**
- Modify: `src/agents/recommender.py`
- Modify: `tests/test_agent_prompts.py`
- Modify if needed for formatting regression only: `tests/test_chatbot.py`

**Interfaces:**
- Consumes: `compose_prompt(task_instruction: str) -> str` from Task 1.
- Produces: natural user-facing recommendation reasons through existing `generate_recommendations(...) -> RecommendationResponse`.

- [ ] **Step 1: Write failing recommendation prompt tests**

Add assertions that the prompt instructs the LLM to:
- recommend only provided candidate IDs;
- produce concise reasons in a professional but approachable consultant style;
- explain fit using cuisine, ingredients, price, location, atmosphere, preparation time, dietary fit, or supported review evidence;
- respond in the same language as the user's request when practical;
- never mention ranking/fusion/modality/semantic scores, embeddings, candidate IDs, agents, or retrieval internals;
- never make unsupported popularity/trend claims;
- avoid padding with weak alternatives merely to fill the list.

Add the Review Focus regression for mixed-language input and internal-score-heavy candidate payloads by capturing the prompt sent to a fake LLM.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest tests/test_agent_prompts.py -k recommender -v`

Expected: FAIL because the current prompt is too generic.

- [ ] **Step 3: Update `src/agents/recommender.py`**

Replace the current short `PROMPT` with the composed production recommendation instruction. Keep candidate-ID validation unchanged.

- [ ] **Step 4: Verify user-facing formatting still caps output at three grounded candidates**

Run: `pytest tests/test_chatbot.py -k "format or retrieve_candidates" -v`

Expected: PASS.

- [ ] **Step 5: Run recommender prompt tests and verify pass**

Run: `pytest tests/test_agent_prompts.py -k recommender -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agents/recommender.py tests/test_agent_prompts.py tests/test_chatbot.py
git commit -m "feat: make recommendations production conversational"
```

### Task 5: Full Regression Verification

**Files:**
- No production files unless a regression discovered here requires a targeted fix.
- Tests: full `tests/` suite.

**Interfaces:**
- Consumes: all prompt changes from Tasks 1-4.
- Produces: verified branch state with no regression to grounding, retrieval, schemas, or workflow behavior.

- [ ] **Step 1: Run prompt-focused tests together**

Run: `pytest tests/test_prompt_policy.py tests/test_agent_prompts.py tests/test_chatbot.py -v`

Expected: PASS.

- [ ] **Step 2: Run the complete test suite**

Run: `pytest -v`

Expected: all tests PASS. If unrelated pre-existing failures exist, record them explicitly rather than hiding them.

- [ ] **Step 3: Run syntax/whitespace verification**

Run: `git diff --check`

Expected: no output.

- [ ] **Step 4: Manually inspect representative prompt outputs with fake/local payloads**

Verify at least these scenarios at the prompt-contract level without paid API calls:
- English restaurant request;
- Vietnamese restaurant request;
- recipe request;
- explicit dietary hard constraint;
- sparse nutrition metadata;
- candidate records containing ranking/fusion score fields.

Expected: prompts require grounded, natural, same-language responses and suppress internal-system terminology.

- [ ] **Step 5: Commit any verification-only test fixes if required**

```bash
git add tests src
git commit -m "test: verify production culinary prompt behavior"
```
