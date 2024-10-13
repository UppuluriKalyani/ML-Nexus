<div align="center">
<img src="https://socialify.git.ci/yashksaini-coder/AI-Query-master/image?font=Raleway&language=1&name=1&theme=Auto" alt="AI-Query-master" width="640" height="320" />
<br><br>
    <img alt="GitHub Repo Name" src="https://img.shields.io/badge/AI_Query_master-2a9d8f">
    <img alt="GitHub Author" src="https://img.shields.io/badge/Author-Yash%20K.%20Saini-778da9">
    <img alt="GitHub commit-activity" src="https://img.shields.io/github/commit-activity/t/yashksaini-coder/AI-Query-master">
    <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/yashksaini-coder/AI-Query-master">
    <img alt="GitHub Created At" src="https://img.shields.io/github/created-at/yashksaini-coder/AI-Query-master">
    <img alt="GitHub Last Commit" src="https://img.shields.io/github/last-commit/yashksaini-coder/AI-Query-master">
    <img alt="GitHub Repo Size" src="https://img.shields.io/github/repo-size/yashksaini-coder/AI-Query-master">
    <img alt="GitHub License" src="https://img.shields.io/github/license/yashksaini-coder/AI-Query-master">
    <img alt="GitHub Open Issues" src="https://img.shields.io/github/issues/yashksaini-coder/AI-Query-master">
    <img alt="GitHub Closed Issues" src="https://img.shields.io/github/issues-closed/yashksaini-coder/AI-Query-master">
    <img alt="GitHub Open PR" src="https://img.shields.io/github/issues-pr/yashksaini-coder/AI-Query-master">
    <img alt="GitHub Closed PR" src="https://img.shields.io/github/issues-pr-closed/yashksaini-coder/AI-Query-master">
    <img alt="GitHub Forks" src="https://img.shields.io/github/forks/yashksaini-coder/AI-Query-master">
    <img alt="GitHub Stars" src="https://img.shields.io/github/stars/yashksaini-coder/AI-Query-master">
    <img alt="GitHub Watchers" src="https://img.shields.io/github/watchers/yashksaini-coder/AI-Query-master">
    <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/yashksaini-coder/AI-Query-master">
</div>
<br>


<div align='center'>
    <a href="mailto:ys3853428@gmail.com"> <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"> </a>
    <a href="https://github.com/yashksaini-coder"> <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"> </a>
    <a href="https://medium.com/@yashksaini"> <img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white"> </a>
    <a href="https://www.linkedin.com/in/yashksaini/"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"> </a>
    <a href="https://bento.me/yashksaini"> <img src="https://img.shields.io/badge/Bento-768CFF.svg?style=for-the-badge&logo=Bento&logoColor=white"> </a>
    <a href="https://www.instagram.com/yashksaini.codes/"> <img src="https://img.shields.io/badge/Instagram-%23FF006E.svg?style=for-the-badge&logo=Instagram&logoColor=white"> </a>
    <a href="https://twitter.com/EasycodesDev"> <img src="https://img.shields.io/badge/X-%23000000.svg?style=for-the-badge&logo=X&logoColor=white"> </a>
</div>
<br>

# AI SQL Query Generator

An advanced AI-powered SQL query generator that converts natural language to optimized SQL queries using Google's Gemini Pro LLM. This Flask-based application supports multiple databases and provides an intuitive interface for query generation and optimization.

## Features

- 🤖 Natural Language to SQL conversion using Gemini Pro
- 🎯 Support for Multiple Databases (PostgreSQL, MySQL)
- ⚡ Real-time Query Validation
- 📊 Query Optimization Suggestions
- 💻 Modern, Responsive UI
- 🛡️ Robust Error Handling
- 🎨 Syntax Highlighting for SQL
- 📝 Detailed Query Explanations

## Prerequisites

- Python 3.8 or higher
- Google Cloud Account with Gemini Pro API access
- PostgreSQL (Neon DB) or MySQL database
- Git (for cloning the repository)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-sql-generator.git
cd ai-sql-generator
```

### 2. Create and Activate Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Neon DB (PostgreSQL)

1. Create a Neon account:
   - Visit [Neon DB](https://neon.tech)
   - Sign up for a free account

2. Create a New Project:
   - Click "Create New Project"
   - Choose a project name
   - Select your region
   - Click "Create Project"

3. Get Connection Details:
   - In your project dashboard, click "Connection Details"
   - Note down the following:
     - Host
     - Database name
     - User
     - Password
   - Your connection string will look like:
     ```
     postgres://[user]:[password]@[host]/[database]
     ```

### 5. Configure Environment Variables

1. Copy the example .env file:
```bash
cp .env.example .env
```

2. Edit `.env` with your credentials:
```plaintext
GOOGLE_API_KEY=your_gemini_pro_api_key_here

# Neon DB Configuration
DB_TYPE=postgresql
DB_HOST=your-neon-db-host.cloud.neon.tech
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=your_secret_key_here
```

### 6. Run the Application

```bash
flask run
```

Visit `http://localhost:8501` in your browser.

## Usage

1. **Enter Database Schema:**
   - Provide your database schema in the designated text area
   - Include table names, columns, and relationships

2. **Write Natural Language Query:**
   - Enter your query in plain English
   - Example: "Show all customers who made purchases over $1000 last month"

3. **Generate SQL:**
   - Click "Generate SQL" button
   - Review the generated query, explanation, and optimization suggestions

4. **Execute Query:**
   - The generated SQL can be copied and executed in your database management tool


## Troubleshooting

### Common Issues

1. **Database Connection Errors:**
   - Verify Neon DB credentials in `.env`
   - Check if IP is whitelisted in Neon DB settings
   - Ensure database server is running

2. **Gemini API Issues:**
   - Verify API key is correct
   - Check API quota and limits
   - Ensure internet connectivity

3. **Application Errors:**
   - Check Streamlit debug logs
   - Verify all dependencies are installed
   - Ensure Python version compatibility

## Contributing

We welcome contributions to enhance the functionality and usability of this project. Please follow these steps to contribute:

### 1. Fork the Repository

Fork the repository to your own GitHub account by clicking the [Fork](yashksaini-coder/AI-Query-master/fork) button at the top right of the repository page.

### 2. Clone Your Fork

Clone your forked repository to your local machine:

```bash
git clone https://github.com/yourusername/ai-sql-generator.git
cd ai-sql-generator
```

### 3. Create a Feature Branch

Create a new branch for your feature or bug fix:

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes

Make the necessary changes to the codebase. Ensure your code follows the project's coding standards and includes appropriate tests.

### 5. Commit Your Changes

Commit your changes with a descriptive commit message:

```bash
git add .
git commit -m "Add feature: your feature description"
```

### 6. Push to Your Branch

Push your changes to your forked repository:

```bash
git push origin feature/your-feature-name
```

### 7. Open a Pull Request

Open a pull request from your feature branch to the main repository's `main` branch. Provide a clear description of your changes and any relevant information.

### 8. Review Process

Your pull request will be reviewed by the maintainers. Please be responsive to any feedback or requests for changes.

### 9. Merge

Once your pull request is approved, it will be merged into the main repository. Congratulations, you have contributed to the project!

Thank you for your contributions!


## License

This project is licensed under the MIT License - see the LICENSE file for details.