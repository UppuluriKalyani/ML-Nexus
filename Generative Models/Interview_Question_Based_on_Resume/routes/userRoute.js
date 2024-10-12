const express = require('express');
const router = express.Router();
const userCtrl = require('../controllers/userCtrl');
const { authUser } = require('../middlewares/authUser');

//Render the login page
router.get('/',userCtrl.loginRender);

//Register the user
router.post('/register', userCtrl.registerUser);

//Login the user
router.post('/login', authUser ,userCtrl.loginUser);

module.exports = router;