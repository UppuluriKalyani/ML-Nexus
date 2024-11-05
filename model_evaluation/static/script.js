// Function to automatically hide flash messages after a few seconds
document.addEventListener("DOMContentLoaded", () => {
    const flashMessages = document.querySelectorAll(".flash-message");
    flashMessages.forEach(msg => {
        setTimeout(() => {
            msg.style.display = "none";
        }, 5000); // Adjust the timeout as needed (5000ms = 5 seconds)
    });
});

// Form validation for dataset upload and model evaluation
document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.querySelector("form[action='/upload']");
    const evaluateForm = document.querySelector("form[action='/evaluate']");
    
    if (uploadForm) {
        uploadForm.addEventListener("submit", (e) => {
            const fileInput = uploadForm.querySelector("input[type='file']");
            if (!fileInput.value) {
                alert("Please select a dataset file to upload.");
                e.preventDefault();
            }
        });
    }

    if (evaluateForm) {
        evaluateForm.addEventListener("submit", (e) => {
            const modelSelect = evaluateForm.querySelector("select[name='model_name']");
            const datasetSelect = evaluateForm.querySelector("select[name='dataset_name']");
            
            if (!modelSelect.value || !datasetSelect.value) {
                alert("Please select both a model and a dataset to evaluate.");
                e.preventDefault();
            }
        });
    }
});
