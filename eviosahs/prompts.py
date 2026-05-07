from __future__ import annotations

from typing import Any


VISUAL_QUESTION_SPECS: list[dict[str, str]] = [
    {
        "anatomy_target": "neck",
        "question": "Measure the thickness of the neck. Is the neck thick or large?",
    },
    {
        "anatomy_target": "chin",
        "question": "Observe the chin. Is the chin receded or pushed back relative to the rest of the face?",
    },
    {
        "anatomy_target": "mouth",
        "question": "Look at the mouth area. Does the mouth appear narrow or crowded?",
    },
    {
        "anatomy_target": "face_and_neck_fat",
        "question": "Look at the face and neck. Is there any sign of excess fatty tissue in these areas?",
    },
    {
        "anatomy_target": "lower_jaw",
        "question": "Look at the lower jaw. Is the jaw small or set back?",
    },
    {
        "anatomy_target": "midface",
        "question": "Observe the midface (area between the eyes and mouth). Does this area appear flat or underdeveloped?",
    },
    {
        "anatomy_target": "nose",
        "question": "Look at the nose. Is there any indication of a deviated septum or nasal obstruction?",
    },
]

SINGLE_VISUAL_QUESTION_SPEC = {
    "anatomy_target": "global_face_airway_risk",
    "question": (
        "Assess the full face and visible upper airway-related anatomy in one pass. "
        "Summarize only OSAHS-relevant anatomical findings involving the neck, chin, mouth, "
        "facial fat distribution, lower jaw, midface, and nose."
    ),
}


VISUAL_SYSTEM_PROMPT = """You are extracting OSAHS-relevant facial anatomy facts from one patient image.
Focus only on the requested anatomy target.
Do not mention clothing, background, pets, accessories, camera angle, lighting, or unrelated objects unless they block visibility.
Do not diagnose OSAHS in this step.
Do not use markdown or code fences."""


REASON_SYSTEM_PROMPT = """You are converting one facial observation into a short OSAHS screening evidence card.
This is a screening task, not a definitive diagnosis task.
Be cautious when visibility is uncertain.
Use the clinical summary to calibrate the importance of the visual finding.
Do not default to supports, against, or uncertain.
Use weak support for mild findings when they add limited risk, rather than forcing them into uncertain.
Do not use markdown or code fences."""


FINAL_SYSTEM_PROMPT = """You are making a binary OSAHS screening decision.
This is a screening label, not a definitive clinical diagnosis.
Use both the evidence cards and the clinical summary.
Do not default to yes or no.
Treat both labels as equally acceptable outcomes.
Base the decision on the balance of supporting versus opposing evidence, plus the structured clinical profile.
Do not default to 'no' only because polysomnography is unavailable.
Use the combined anatomical evidence and the clinical summary together.
Do not use markdown or code fences."""


def build_visual_prompt(question: str, anatomy_target: str) -> str:
    return f"""Question: {question}
Target anatomy: {anatomy_target}

Rules:
1. Only describe the requested anatomy target.
2. Ignore clothing, background, accessories, pets, medical devices, and non-anatomical objects unless they block visibility.
3. If the anatomy is not visible enough, explicitly mark visibility as uncertain.
4. Do not mention diagnosis, treatment, or sleep tests.

Return exactly these 3 lines:
AnatomyTarget: <{anatomy_target}>
VisualObservation: <one short sentence about the target anatomy only>
Visibility: <high|medium|uncertain>"""


def build_reason_prompt(
    anatomy_target: str,
    visual_observation: str,
    visibility: str,
    clinical_summary: str,
    *,
    include_clinical_summary: bool = True,
    reason_style: str = "react",
    include_evidence_strength: bool = True,
) -> str:
    clinical_block = clinical_summary if include_clinical_summary else "Not provided at this stage."
    if reason_style not in {"react", "summary"}:
        raise ValueError(f"Unsupported reason_style: {reason_style}")

    shared_rules = [
        "1. Focus on airway or craniofacial screening relevance in the context of the available evidence.",
        "2. Do not default to supports, against, or uncertain.",
        "3. Use RiskDirection: against only when the visual finding is genuinely reassuring or argues against screening risk.",
        "4. If visibility is uncertain, be conservative and prefer RiskDirection: uncertain unless the observation is still clearly informative.",
        "5. Use the clinical summary to calibrate the significance of the visual finding when it is available at this stage.",
        "6. Do not give a final yes/no diagnosis in this step.",
        "7. Do not mention polysomnography, diagnosis thresholds, or differential diagnosis.",
        "8. Keep each field short and concrete.",
    ]
    if include_evidence_strength:
        shared_rules.insert(
            2,
            "3. If a finding is mild but still relevant, you may use RiskDirection: supports with EvidenceStrength: weak instead of forcing it to uncertain.",
        )

    if reason_style == "summary":
        output_lines = [
            "Observation: <repeat the key anatomical observation in a precise way>",
            "Interpretation: <clinical meaning for OSAHS screening in this patient>",
            "RiskDirection: <supports|against|uncertain>",
        ]
        if include_evidence_strength:
            output_lines.append("EvidenceStrength: <weak|moderate|strong>")
        output_lines.extend(
            [
                "Confidence: <high|medium|low>",
                "EvidenceSummary: <one short sentence for final aggregation>",
            ]
        )
        return f"""Target anatomy: {anatomy_target}
Visual observation: {visual_observation}
Visibility: {visibility}
Clinical summary: {clinical_block}

Task:
Produce a compact evidence card for OSAHS screening without using ReAct-style intermediate steps.

Rules:
{chr(10).join(shared_rules)}

Return exactly these {len(output_lines)} lines:
{chr(10).join(output_lines)}"""

    output_lines = [
        "Thought: <one short reasoning sentence>",
        "Action: <what screening relevance you are evaluating>",
        "Observation: <repeat the key anatomical observation in a precise way, with clinical context if relevant>",
        "Interpretation: <clinical meaning for OSAHS screening in this patient>",
        "FinalThought: <short final thought about how much this feature should influence screening risk>",
        "RiskDirection: <supports|against|uncertain>",
    ]
    if include_evidence_strength:
        output_lines.append("EvidenceStrength: <weak|moderate|strong>")
    output_lines.extend(
        [
            "Confidence: <high|medium|low>",
            "EvidenceSummary: <one short sentence for final aggregation>",
        ]
    )
    return f"""Target anatomy: {anatomy_target}
Visual observation: {visual_observation}
Visibility: {visibility}
Clinical summary: {clinical_block}

Task:
Use a short ReAct-style reasoning trace to assess how this single anatomical observation should be interpreted for OSAHS screening in the current clinical context.

Rules:
{chr(10).join(shared_rules)}

Return exactly these {len(output_lines)} lines:
{chr(10).join(output_lines)}"""


def format_evidence_cards(session: list[dict[str, Any]], *, include_evidence_strength: bool = True) -> str:
    lines: list[str] = []
    for index, item in enumerate(sorted(session, key=lambda value: int(value.get("session_index", 0))), start=1):
        card = item.get("evidence_card", {})
        block = [
            f"Session {index}",
            f"AnatomyTarget: {item.get('anatomy_target', '')}",
            f"Visibility: {item.get('visibility', '')}",
            f"RiskDirection: {card.get('risk_direction', '')}",
        ]
        if include_evidence_strength:
            block.append(f"EvidenceStrength: {card.get('evidence_strength', '')}")
        block.extend(
            [
                f"Confidence: {card.get('confidence', '')}",
                f"Observation: {card.get('observation') or item.get('visual_observation', '')}",
                f"FinalThought: {card.get('final_thought', '')}",
                f"EvidenceSummary: {card.get('evidence_summary', '')}",
            ]
        )
        lines.append("\n".join(block))
    return "\n\n".join(lines)


def build_final_prompt(
    evidence_cards_text: str,
    semantic_text: str,
    *,
    balanced: bool = True,
    include_evidence_strength: bool = True,
) -> str:
    decision_rules = [
        "1. First assess the balance of supports, against, and uncertain evidence.",
        "2. Then assess whether the structured clinical summary overall leans low, medium, or high risk.",
    ]
    if include_evidence_strength:
        decision_rules[0] = "1. First assess the balance of supports, against, and uncertain evidence, paying attention to evidence strength."
    if balanced:
        decision_rules.extend(
            [
                "3. Do not default to yes or no.",
                "4. Output yes only when the combined evidence balance and clinical profile lean positive overall.",
                "5. Output no only when the combined evidence balance and clinical profile lean negative overall.",
                "6. When the case is mixed, use the stronger evidence and the structured clinical profile to break the tie, without defaulting to either label.",
                "7. Do not default to no only because polysomnography is unavailable.",
            ]
        )
        output_lines = [
            "SupportCount: <int>",
            "AgainstCount: <int>",
            "UncertainCount: <int>",
            "ClinicalRiskLevel: <low|medium|high>",
            "BriefReasoning: <1 to 3 short sentences>",
            "Final answer: <yes|no>",
        ]
    else:
        decision_rules.extend(
            [
                "3. Use the overall evidence pattern and the clinical profile to make a direct screening decision.",
                "4. Keep the reasoning concise.",
            ]
        )
        output_lines = [
            "BriefReasoning: <1 to 3 short sentences>",
            "Final answer: <yes|no>",
        ]
    return f"""This is a binary OSAHS screening task.
Decide whether the patient should be considered screening-positive for OSAHS.

Evidence cards:
{evidence_cards_text}

Clinical summary:
{semantic_text}

Decision rules:
{chr(10).join(decision_rules)}

Return exactly these {len(output_lines)} lines:
{chr(10).join(output_lines)}"""


def build_direct_multimodal_prompt(clinical_summary: str, *, balanced: bool = True) -> str:
    output_lines = [
        "BriefReasoning: <1 to 3 short sentences>",
        "Final answer: <yes|no>",
    ]
    if balanced:
        output_lines = [
            "SupportCount: <int>",
            "AgainstCount: <int>",
            "UncertainCount: <int>",
            "ClinicalRiskLevel: <low|medium|high>",
            "BriefReasoning: <1 to 3 short sentences>",
            "Final answer: <yes|no>",
        ]
    return f"""This is a binary OSAHS screening task based on a single face image and a structured clinical summary.
Decide whether the patient should be considered screening-positive for OSAHS.

Clinical summary:
{clinical_summary}

Rules:
1. Use only visible OSAHS-relevant anatomical cues from the image.
2. Use the structured clinical summary together with the image findings.
3. Do not default to yes or no.
4. This is a screening decision, not a definitive diagnosis.
5. Do not rely on clothing, background, accessories, or unrelated objects.

Return exactly these {len(output_lines)} lines:
{chr(10).join(output_lines)}"""


def build_clinical_only_prompt(clinical_summary: str, *, balanced: bool = True) -> str:
    output_lines = [
        "BriefReasoning: <1 to 3 short sentences>",
        "Final answer: <yes|no>",
    ]
    if balanced:
        output_lines = [
            "ClinicalRiskLevel: <low|medium|high>",
            "BriefReasoning: <1 to 3 short sentences>",
            "Final answer: <yes|no>",
        ]
    return f"""This is a binary OSAHS screening task using only structured clinical information.
Decide whether the patient should be considered screening-positive for OSAHS.

Clinical summary:
{clinical_summary}

Rules:
1. Use only the clinical summary. Do not infer unavailable image findings.
2. Do not default to yes or no.
3. This is a screening decision, not a definitive diagnosis.

Return exactly these {len(output_lines)} lines:
{chr(10).join(output_lines)}"""
