const inputType = document.getElementById("inputType");
const textInputGroup = document.getElementById("textInputGroup");
const fileInputGroup = document.getElementById("fileInputGroup");
const method = document.getElementById("method");
const methodNote = document.getElementById("methodNote");

const summarizeButton = document.getElementById("summarizeButton");
const rougeButton = document.getElementById("rougeButton");
const downloadButton = document.getElementById("downloadButton");

const loading = document.getElementById("loading");
const warningBox = document.getElementById("warningBox");
const summaryOutput = document.getElementById("summaryOutput");
const riskContainer = document.getElementById("riskContainer");

const originalWordCount = document.getElementById("originalWordCount");
const summaryWordCount = document.getElementById("summaryWordCount");
const detectedLanguage = document.getElementById("detectedLanguage");
const usedModel = document.getElementById("usedModel");

inputType.addEventListener("change", function () {
    if (inputType.value === "text") {
        textInputGroup.classList.remove("hidden");
        fileInputGroup.classList.add("hidden");
    } else {
        textInputGroup.classList.add("hidden");
        fileInputGroup.classList.remove("hidden");
    }
});

method.addEventListener("change", function () {
    if (method.value === "bart") {
        methodNote.textContent = "BART is recommended for English Terms and Conditions documents. It creates an abstractive rewritten summary.";
    } else {
        methodNote.textContent = "TF-IDF is recommended for Indonesian, English, or mixed text. It extracts the most important original sentences.";
    }
});

summarizeButton.addEventListener("click", async function () {
    const formData = new FormData();

    const selectedInputType = document.getElementById("inputType").value;
    const selectedMethod = document.getElementById("method").value;

    formData.append("inputType", selectedInputType);
    formData.append("method", selectedMethod);

    if (selectedInputType === "text") {
        const text = document.getElementById("textInput").value;
        formData.append("text", text);
    } else {
        const file = document.getElementById("fileInput").files[0];

        if (!file) {
            alert("Please upload a PDF or TXT file first.");
            return;
        }

        formData.append("file", file);
    }

    loading.classList.remove("hidden");
    warningBox.classList.add("hidden");
    warningBox.textContent = "";

    summaryOutput.value = "";
    riskContainer.innerHTML = "";

    originalWordCount.textContent = "0";
    summaryWordCount.textContent = "0";
    detectedLanguage.textContent = "-";
    usedModel.textContent = "-";

    try {
        const response = await fetch("/summarize", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        loading.classList.add("hidden");

        if (!data.success) {
            alert(data.message);
            return;
        }

        summaryOutput.value = data.summary;
        originalWordCount.textContent = data.originalWordCount;
        summaryWordCount.textContent = data.summaryWordCount;
        detectedLanguage.textContent = data.language;
        usedModel.textContent = data.modelName;

        if (data.warning && data.warning.trim() !== "") {
            warningBox.textContent = data.warning;
            warningBox.classList.remove("hidden");
        }

        displayRisks(data.risks);

    } catch (error) {
        loading.classList.add("hidden");
        alert("Something went wrong while processing the document.");
        console.log(error);
    }
});

function displayRisks(risks) {
    riskContainer.innerHTML = "";

    if (!risks || risks.length === 0) {
        riskContainer.innerHTML = `
            <p class="empty-message">
                No common legal risk keywords were detected.
            </p>
        `;
        return;
    }

    risks.forEach(function (risk) {
        const card = document.createElement("div");
        card.className = "risk-card";

        card.innerHTML = `
            <h3>${risk.category}</h3>
            <p><strong>Detected keywords:</strong></p>
            <p>${risk.keywords.join(", ")}</p>
        `;

        riskContainer.appendChild(card);
    });
}

downloadButton.addEventListener("click", function () {
    const summary = summaryOutput.value;

    if (!summary) {
        alert("No summary available to download.");
        return;
    }

    const blob = new Blob([summary], {
        type: "text/plain"
    });

    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "summarly_legal_summary.txt";
    link.click();
});

rougeButton.addEventListener("click", async function () {
    const reference = document.getElementById("referenceSummary").value;
    const generated = summaryOutput.value;
    const rougeResult = document.getElementById("rougeResult");

    if (!reference || !generated) {
        alert("Please provide both reference summary and generated summary.");
        return;
    }

    try {
        const response = await fetch("/rouge", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                reference: reference,
                generated: generated
            })
        });

        const data = await response.json();

        if (!data.success) {
            alert(data.message);
            return;
        }

        const scores = data.scores;

        rougeResult.innerHTML = `
            <table class="rouge-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>ROUGE-1</td>
                        <td>${scores.rouge1.precision}</td>
                        <td>${scores.rouge1.recall}</td>
                        <td>${scores.rouge1.f1}</td>
                    </tr>
                    <tr>
                        <td>ROUGE-2</td>
                        <td>${scores.rouge2.precision}</td>
                        <td>${scores.rouge2.recall}</td>
                        <td>${scores.rouge2.f1}</td>
                    </tr>
                    <tr>
                        <td>ROUGE-L</td>
                        <td>${scores.rougeL.precision}</td>
                        <td>${scores.rougeL.recall}</td>
                        <td>${scores.rougeL.f1}</td>
                    </tr>
                </tbody>
            </table>
        `;

    } catch (error) {
        alert("Failed to calculate ROUGE score.");
        console.log(error);
    }
});