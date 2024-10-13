document.getElementById('search-form').addEventListener('submit', function (event) {
    event.preventDefault();
    
    let query = document.getElementById('query').value;
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'query': query
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        displayResults(data);
        displayChart(data);
    });
});

function displayResults(data) {
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<h2>Results</h2>';
    for (let i = 0; i < data.documents.length; i++) {
        let docDiv = document.createElement('div');
        docDiv.innerHTML = `<strong>Document ${data.indices[i]}</strong><p>${data.documents[i]}</p><br><strong>Similarity: ${data.similarities[i]}</strong>`;
        resultsDiv.appendChild(docDiv);
    }
}

// Define a variable to hold the chart instance
let similarityChart = null;

function displayChart(data) {
    let ctx = document.getElementById('similarity-chart').getContext('2d');
    
    // If a chart already exists, destroy it first
    if (similarityChart) {
        similarityChart.destroy();
    }

    // Create a new chart
    similarityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.indices.map(index => `Document ${index}`),
            datasets: [{
                label: 'Cosine Similarity',
                data: data.similarities,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Similarity Score'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Document Index'
                    }
                }
            },
            responsive: true
        }
    });
}
