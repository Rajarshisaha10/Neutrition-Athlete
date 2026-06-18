import re

with open('c:/Users/rajar/Desktop/coding/temp_repo/app/frontend/app.js', 'r') as f:
    content = f.read()

content = content.replace('const API_BASE = window.ATHLETEEDGE_API_URL || "";', 'const API_BASE = window.ATHLETEEDGE_API_URL || "/api";')

content = re.sub(r'\{ key: "sleep_quality".*?\},', '{ key: "sleep_quality", title: "Sleep quality", category: "Recovery inputs", type: "select", options: ["1 (Almost no sleep)", "2", "3 (Okay sleep)", "4", "5 (Deep full sleep)"], value: "3 (Okay sleep)", help: "Rate last night\'s sleep 1 to 5.", guide: "1 is barely slept, 5 is deeply rested." },', content)
content = re.sub(r'\{ key: "recovery_score".*?\},', '{ key: "recovery_score", title: "Recovery score", category: "Recovery inputs", type: "select", options: ["1 (Very poor)", "2", "3 (Average)", "4", "5 (Excellent)"], value: "3 (Average)", help: "Rate how ready the body feels 1 to 5.", guide: "1 is heavy/pain, 5 is fresh/energetic." },', content)
content = re.sub(r'\{ key: "stress_level".*?\},', '{ key: "stress_level", title: "Stress level", category: "Readiness inputs", type: "select", options: ["1 (Calm)", "2", "3 (Normal pressure)", "4", "5 (Very tense)"], value: "3 (Normal pressure)", help: "Rate current mental/physical stress 1 to 5.", guide: "1 is calm, 5 is overloaded or worried." },', content)
content = re.sub(r'\{ key: "training_intensity".*?\},', '{ key: "training_intensity", title: "Training intensity", category: "Workload inputs", type: "select", options: ["1 (Very easy)", "2", "3 (Moderate)", "4", "5 (Very hard)"], value: "3 (Moderate)", help: "How hard was the session 1 to 5?", guide: "1 is light, 5 is match-level intensity." },', content)
content = re.sub(r'\{ key: "fatigue_index".*?\},', '{ key: "fatigue_index", title: "Fatigue index", category: "Readiness inputs", type: "select", options: ["1 (Fresh)", "2", "3 (Heavy legs)", "4", "5 (Exhausted)"], value: "3 (Heavy legs)", help: "Rate how tired you feel 1 to 5.", guide: "1 is completely fresh, 5 is exhausted." },', content)
content = re.sub(r'\s*\{ key: "training_load".*?\},', '', content)

old_logic = '''function riskPayload() {
  return { ...riskValues };
}'''

new_logic = '''function parse1to5(value, maxVal) {
  if (typeof value === 'string') {
    const match = value.match(/^(\\d)/);
    if (match) {
      const num = parseInt(match[1], 10);
      return (num / 5) * maxVal;
    }
  }
  return Number(value) || 0;
}

function riskPayload() {
  const payload = { ...riskValues };
  payload.sleep_quality = parse1to5(payload.sleep_quality, 10);
  payload.recovery_score = parse1to5(payload.recovery_score, 100);
  payload.stress_level = parse1to5(payload.stress_level, 1.0);
  payload.training_intensity = parse1to5(payload.training_intensity, 10);
  payload.fatigue_index = parse1to5(payload.fatigue_index, 100);
  payload.training_load = payload.training_intensity * Number(payload.training_duration || 0);
  return payload;
}'''

content = content.replace(old_logic, new_logic)

with open('c:/Users/rajar/Desktop/coding/AthleteEdge AI/static/injuries.js', 'w') as f:
    f.write(content)
print("Done")
