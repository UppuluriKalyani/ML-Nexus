const express = require('express');
const router = express.Router();
const mainCtrl = require('../controllers/mainCtrl')

//Routes
router.get('/',mainCtrl.homepage);
router.get('/about',mainCtrl.about);

module.exports = router