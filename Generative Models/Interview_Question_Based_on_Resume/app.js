const express = require('express');
const expressLayouts = require('express-ejs-layouts');
const dbConnect = require('./config/dbConnect');
require('dotenv').config();

const app = express();
const PORT =  process.env.PORT || 5000;

dbConnect(); // Connect to MongoDB

app.use(express.urlencoded({extended: true}));
app.use(express.json());

app.use(express.static('public'));

//Templating Engine
app.use(expressLayouts);
app.set('layout','./layouts/main');
app.set('view engine', 'ejs');

//Routes
app.use('/',require('./routes/indexRoute'));
app.use('/dashboard',require('./routes/dashboardRoute'));
app.use('/auth',require('./routes/userRoute'));

//Handle Not Found - 404
app.get('*', function(req, res) {
    res.status(404).render('404');
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});