const fs = require('fs');
const TurndownService = require('turndown');

function convertHtmlToMarkdown(inputFilePath, outputFilePath) {
    fs.readFile(inputFilePath, 'utf8', (err, data) => {
        if (err) {
            console.error('Error reading the file:', err);
            return;
        }

        const turndownService = new TurndownService();
        const markdown = turndownService.turndown(data);

        fs.writeFile(outputFilePath, markdown, (err) => {
            if (err) {
                console.error('Error writing the file:', err);
                return;
            }

            console.log('Markdown file has been saved.');
        });
    });
}

// Example usage
convertHtmlToMarkdown('index.html', 'output.md');