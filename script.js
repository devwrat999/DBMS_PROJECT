// Wait for the DOM to be fully loaded before running script
document.addEventListener('DOMContentLoaded', () => {

    // --- Get DOM Elements ---
    const raForm = document.getElementById('ra-form');
    const raInput = document.getElementById('ra-input');
    const sqlOutput = document.getElementById('sql-output');
    const convertBtn = document.getElementById('convert-btn');
    const notification = document.getElementById('notification');

    // --- Define API URL ---
    // This script assumes the Python server is running on localhost port 5000
    const API_URL = 'http://127.0.0.1:5000/convert';

    // --- SQL Keywords for Highlighting ---
    const SQL_KEYWORDS = [
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'GROUP BY', 'HAVING',
        'UNION', 'INTERSECT', 'EXCEPT', 'AS', 'CROSS', 'INNER', 'LEFT', 'RIGHT',
        'ORDER BY', 'LIMIT', 'OFFSET', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'AND', 'OR', 'NOT'
    ];
    // Create a regex to find all keywords (case-insensitive, word boundaries)
    const keywordRegex = new RegExp(`\\b(${SQL_KEYWORDS.join('|')})\\b`, 'gi');
    const literalRegex = new RegExp(`('.*?')`, 'g');

    /**
     * Applies syntax highlighting to the raw SQL string and returns HTML.
     * @param {string} sql - The raw SQL query.
     * @returns {string} - HTML string with <span> tags for highlighting.
     */
    function highlightSQL(sql) {
        // We use a safe replacement to avoid issues with special regex characters
        let html = sql
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
            
        // Highlight keywords
        html = html.replace(keywordRegex, (match) => {
            return `<span class="sql-keyword">${match}</span>`;
        });
        
        // Highlight literals (e.g., 'CSE')
        html = html.replace(literalRegex, (match) => {
            return `<span class="sql-literal">${match}</span>`;
        });

        return html;
    }

    /**
     * Shows the notification banner with a message.
     * @param {string} message - The text to display.
     * @param {boolean} isError - If true, show the error style.
     */
    function showNotification(message, isError = false) {
        notification.textContent = message;
        notification.className = isError ? 'error' : 'success';
        
        // Hide notification after 3 seconds
        setTimeout(() => {
            notification.className = '';
        }, 3000);
    }

    // --- Form Submit Event Handler ---
    raForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Stop the form from reloading the page
        
        const expression = raInput.value.trim();
        if (!expression) {
            showNotification('Please enter an expression.', true);
            return;
        }

        // Disable button to prevent multiple submissions
        convertBtn.disabled = true;
        convertBtn.textContent = 'Converting...';
        notification.className = ''; // Clear previous notification

        try {
            // --- Make the API Call to the Python Backend ---
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ expression: expression }),
            });

            const data = await response.json();

            if (!response.ok) {
                // Handle errors from the server (e.g., bad RA syntax)
                const errorMsg = data.error || 'An unknown error occurred.';
                sqlOutput.innerHTML = `<span style="color: var(--error-text);">${errorMsg}</span>`;
                showNotification('Conversion Failed!', true);
            } else {
                // --- Success ---
                const highlighted = highlightSQL(data.sql);
                sqlOutput.innerHTML = highlighted; // Set the highlighted HTML
                showNotification('Successfully Converted!', false);
            }

        } catch (error) {
            // Handle network errors (e.g., server is not running)
            console.error('Fetch Error:', error);
            const errorMsg = 'Error: Could not connect to the backend server. Is it running?';
            sqlOutput.innerHTML = `<span style="color: var(--error-text);">${errorMsg}</span>`;
            showNotification('Connection Error!', true);
        } finally {
            // Re-enable the button
            convertBtn.disabled = false;
            convertBtn.textContent = '▶ Convert to SQL';
        }
    });
});