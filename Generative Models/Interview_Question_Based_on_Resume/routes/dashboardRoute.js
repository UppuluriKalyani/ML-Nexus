const express = require('express');
const router = express.Router();
const dashboardCtrl = require('../controllers/dashboardCtrl');
const multer = require('multer');
const pdfParse = require('pdf-parse');
const fs = require('fs').promises;  
const path = require('path');
const axios = require('axios');
const { GoogleGenerativeAI } = require("@google/generative-ai");
const genAI = new GoogleGenerativeAI(process.env.API_KEY);

//Routes
router.get('/',dashboardCtrl.dashboard);

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/')  // Specify the directory where uploaded files will be stored
    },
    filename: function (req, file, cb) {
        cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname)) // Generate unique filename
    }
});

const upload = multer({ storage: storage });

router.post('/', upload.single('resume'), async (req, res) => {
    try {
        console.log('Hi');
        const filePath = path.join(__dirname, '../uploads',req.file.filename);
        const dataBuffer = await fs.readFile(filePath);
        const pdfData = await pdfParse(dataBuffer);

        const pdfText = pdfData.text;
        console.log('Two');
        const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash"});
        const header = "Provide me with 30 interview question and answers for the below resume data also suggest some changes in the resume and provide me the final ATS Score out of 100\n ";
        const prompt = header+pdfText;
        const result = await model.generateContent(prompt);
        const response = await result.response;
        const questions = response.text();
        console.log('Three');
        
        const locals = {
            title : 'Get Questions',
            description: 'Interview',
        };

        res.render('dashboard/questions', {
            locals,
            questions,
            layout: '../views/layouts/dashboard'
        });
    } 
    catch (error) {
        console.error('Error processing upload : ', error);
        res.status(500).send('Error processing upload');    
    }
});

router.post('/skill', dashboardCtrl.registerSkill);

router.post('/search', dashboardCtrl.searchSkill);

module.exports = router;