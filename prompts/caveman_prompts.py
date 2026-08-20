PROMPTS = [ 
"Why is my React component re-rendering on every state update even though the props haven't changed? I'm passing an object as a prop.",
"My Express auth middleware is letting expired JWT tokens through. The expiry check uses Date.now() compared to the token's exp field. What's wrong and how do I fix it?",
"How do I set up a PostgreSQL connection pool in Node.js with proper timeout and error handling configuration?",
"Explain the difference between git rebase and git merge. When should I use each one and what are the tradeoffs?",
"Refactor this callback-based Node.js function to use async/await:\n\nfunction getUser(id, callback) {\n  db.query('SELECT * FROM users WHERE id = ?', [id], function(err, rows) {\n    if (err) return callback(err);\n    if (!rows.length) return callback(new Error('Not found'));\n    callback(null, rows[0]);\n  });\n}",
"We have a monolithic Django app that's getting slow. The team is debating microservices. What are the key factors to consider before splitting up the monolith?",
"Review this Express route handler for security issues:\n\napp.get('/api/users/:id', (req, res) => {\n  const query = SELECT * FROM users WHERE id = ${req.params.id};\n  db.query(query).then(user => res.json(user));\n});",
"Write a multi-stage Dockerfile for a Node.js TypeScript application that minimizes the final image size. The app uses npm and needs to compile TypeScript before running.",
"My Node.js API endpoint that increments a counter in PostgreSQL sometimes returns the same value for concurrent requests. How do I fix this race condition?",
"Implement a React error boundary component that catches render errors, shows a fallback UI with a retry button, and logs the error details."
]
