document.addEventListener('DOMContentLoaded', () => {

    // ── Questions mapped exactly to model features ───────────────────────────
    const questions = [
        { key: 'A1_Score',  text: 'Difficulty making eye contact?' },
        { key: 'A2_Score',  text: "Trouble understanding others' feelings?" },
        { key: 'A3_Score',  text: 'Prefers being alone rather than with others?' },
        { key: 'A4_Score',  text: 'Uncomfortable in social situations?' },
        { key: 'A5_Score',  text: 'Difficulty starting or continuing conversations?' },
        { key: 'A6_Score',  text: 'Takes things very literally?' },
        { key: 'A7_Score',  text: 'Gets overly focused on one topic or interest?' },
        { key: 'A8_Score',  text: 'Upset by small changes in routine?' },
        { key: 'A9_Score',  text: 'Difficulty understanding social rules?' },
        { key: 'A10_Score', text: 'Struggles to express emotions clearly?' },
    ];

    const container   = document.getElementById('questionsContainer');
    const form        = document.getElementById('predictionForm');
    const submitBtn   = document.getElementById('submitBtn');
    const loading     = document.getElementById('loading');
    const resultArea  = document.getElementById('resultArea');

    // ── Inject screening questions ───────────────────────────────────────────
    questions.forEach((q, i) => {
        container.insertAdjacentHTML('beforeend', `
            <div class="question-group">
                <div class="question-text">
                    <p><strong>${i + 1}.</strong> ${q.text}</p>
                </div>
                <div class="toggle-group">
                    <input type="radio" id="${q.key}_yes" name="${q.key}" value="1" required>
                    <label for="${q.key}_yes">Yes</label>
                    <input type="radio" id="${q.key}_no"  name="${q.key}" value="0" checked>
                    <label for="${q.key}_no">No</label>
                </div>
            </div>`);
    });

    // ── Populate dynamic dropdowns ───────────────────────────────────────────
    fetch('/api/options')
        .then(r => r.json())
        .then(data => {
            populate('ethnicity',       data.ethnicity,      'Others');
            populate('contry_of_res',   data.contry_of_res,  'United States');
            populate('relation',        data.relation,       'Self');
        })
        .catch(err => console.error('Options error:', err));

    function populate(id, options, preferred) {
        const sel = document.getElementById(id);
        sel.innerHTML = '';
        options.forEach(opt => {
            const o = document.createElement('option');
            o.value = opt; o.textContent = opt;
            if (opt === preferred) o.selected = true;
            sel.appendChild(o);
        });
    }

    // ── Form submit ──────────────────────────────────────────────────────────
    form.addEventListener('submit', async e => {
        e.preventDefault();
        resultArea.style.display = 'none';
        submitBtn.style.display  = 'none';
        loading.style.display    = 'block';

        const fd   = new FormData(form);
        const data = Object.fromEntries(fd.entries());
        questions.forEach(q => { data[q.key] = parseInt(data[q.key]); });

        try {
            const res    = await fetch('/api/predict', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify(data),
            });
            const result = await res.json();
            if (result.status === 'success') {
                displayResult(result);
            } else {
                alert('Error: ' + result.message);
                submitBtn.style.display = 'block';
            }
        } catch (err) {
            alert('Network error during prediction.');
            console.error(err);
            submitBtn.style.display = 'block';
        } finally {
            loading.style.display = 'none';
        }
    });

    // ── Render results ───────────────────────────────────────────────────────
    function displayResult({ risk_level, asd_probability, message, explanations, recommendations }) {
        // Section 1 – Prediction card
        const card  = document.getElementById('resultCard');
        const badge = document.getElementById('resultBadge');
        const RISK = {
            Low:    { cls: 'low-risk',    icon: '🟢', label: 'Low Risk'    },
            Medium: { cls: 'medium-risk', icon: '🟡', label: 'Medium Risk' },
            High:   { cls: 'high-risk',   icon: '🔴', label: 'High Risk'   },
        };
        const r = RISK[risk_level] || RISK.Low;
        card.className = 'result-card ' + r.cls;
        badge.textContent = r.icon + ' ' + r.label;
        document.getElementById('resultTitle').textContent   = message;
        document.getElementById('resultMessage').textContent =
            `The model assessed an ASD probability of ${asd_probability.toFixed(1)}%.`;

        // Probability bar
        const probPct = Math.min(100, asd_probability).toFixed(1);
        document.getElementById('probValue').textContent = probPct;
        const bar = document.getElementById('probBar');
        bar.style.width = probPct + '%';
        bar.className   = 'prob-bar-fill ' + r.cls.replace('-risk', '-bar');

        // Section 2 – Explanations
        const expList = document.getElementById('explanationList');
        expList.innerHTML = '';
        explanations.forEach(exp => {
            expList.insertAdjacentHTML('beforeend',
                `<li class="exp-item"><span class="exp-dot"></span>${exp}</li>`);
        });

        // Section 3 – Recommendations
        const recoList = document.getElementById('recommendationList');
        recoList.innerHTML = '';

        // Group by trait
        const grouped = {};
        recommendations.forEach(({ trait, tip }) => {
            if (!grouped[trait]) grouped[trait] = [];
            grouped[trait].push(tip);
        });

        Object.entries(grouped).forEach(([trait, tips]) => {
            const block = document.createElement('div');
            block.className = 'reco-block';
            block.innerHTML = `
                <div class="reco-trait">${trait}</div>
                <ul class="reco-tips">
                    ${tips.map(t => `<li>${t}</li>`).join('')}
                </ul>`;
            recoList.appendChild(block);
        });

        form.style.display       = 'none';
        resultArea.style.display = 'block';
    }
});

// ── Reset ────────────────────────────────────────────────────────────────────
function resetForm() {
    document.getElementById('predictionForm').style.display = 'block';
    document.getElementById('predictionForm').reset();
    document.getElementById('resultArea').style.display     = 'none';
    document.getElementById('submitBtn').style.display      = 'block';
    // Reset radio defaults to "No"
    document.querySelectorAll('input[type="radio"][value="0"]')
        .forEach(r => r.checked = true);
}
