// import { Link } from "react-router-dom";
// import Facebook from "./Icons/Facebook";
// import Twitter from "./Icons/Twitter";
// import Dribbble from "./Icons/Dribbble";
// import LinkedIn from "./Icons/LinkedIn";
// import './CSS/Footer.css'
// const Footer = () => {
//     return ( 
//         <>

// <footer class="site-footer">
// <div class="footer5">
// <div class="container">
// <div class="row">
// <div class="col-sm-12 col-md-6">
// <h6 class="text-justify" >About</h6>
// <p class="text-justify">Our platform provides comprehensive summaries of research papers, enabling users to quickly grasp the key points and findings. In addition to summaries, users can also ask questions related to the research papers, facilitating a deeper understanding and discussion of the content.</p>
// </div>
// <div class="col-6 col-md-3">
// <h6>Categories</h6>
// <ul class="footer-links ">
// <Link to="./"><li>Add any</li></Link>
// <Link to="./"><li>Links</li></Link>


// </ul>
// </div>
// <div class="col-6 col-md-3">
// <h6>Quick Links</h6>
// <ul class="footer-links">
// <Link to="./about"><li>About Us</li></Link>
// <div class="button-container">
//   <button class="footer-return-btn js-return-top btn--beta">Back to Top</button>
// </div>

// <Link to="./cont"><li>Contact Us</li></Link>
// <Link to="./serv"><li>Services</li></Link>



// </ul>
// </div>
// </div>
// <hr class="small"></hr>
// </div>
// <div class="container">
// <div class="row">
// <div class="col-md-8 col-sm-6 col-12">
// <p class="copyright-text">Copyright © 2024 All Rights Reserved by
// <Link to="/"><span class="logo">RESEARCH PAPER</span></Link>
// </p>
// </div>
// <div class="col-md-4 col-sm-6 col-12">
// <ul class="social-icons">
// <li><Link class="facebook" to="https://www.facebook.com/"> <Facebook/> <i class="fab fa-facebook-f"></i></Link></li>
// <li><Link class="twitter" to="https://twitter.com/i/flow/login"> <Twitter/> <i class="fab fa-twitter"></i></Link></li>
// <li><Link class="dribbble" to="https://dribbble.com/session/new" > <Dribbble/> <i class="fab fa-dribbble"></i></Link></li>
// <li><Link class="linkedin" to="https://www.linkedin.com/login"> <LinkedIn/> <i class="fab fa-linkedin-in"></i></Link></li>
// </ul>

// </div>
// </div>
// </div>
// </div>
// </footer>

        
        
        
//         </>

//      );
// }
 
// export default Footer;

// src/components/Footer.js
import { Link } from "react-router-dom";
import Facebook from "./Icons/Facebook";
import Twitter from "./Icons/Twitter";
import Dribbble from "./Icons/Dribbble";
import LinkedIn from "./Icons/LinkedIn";
import ScrollToTop from './ScrollToTop';
import './CSS/Footer.css';

const Footer = () => {
  return (
    <>
    
     <div className="button-container1">
                    <button className="footer-return-btn js-return-top btn--beta" style={{marginTop:'150px',backgroundColor:'black'}}>Back to Top</button>
                  </div>
                 
      <footer className="site-footer">
        <div className="footer5">
          <div className="container">
            <div className="row">
              <div className="col-sm-12 col-md-6">
                <h6 className="text-justify" style={{display: 'inline-block'}}>About</h6>
               
                <p className="text-justify">
                  Our platform provides comprehensive summaries of research papers, enabling users to quickly grasp the key points and findings. In addition to summaries, users can also ask questions related to the research papers, facilitating a deeper understanding and discussion of the content.
                </p>
              </div>
              <div className="col-6 col-md-3">
                <h6>Categories</h6>
                <ul className="footer-links">
                  <Link to="./"><li>Add any</li></Link>
                  <Link to="./"><li>Links</li></Link>
                </ul>
              </div>
              <div className="col-6 col-md-3">
                <h6>Quick Links</h6>
                <ul className="footer-links">
                  <Link to="./about"><li>About Us</li></Link>
                  
                  <Link to="./cont"><li>Contact Us</li></Link>
                  <Link to="./serv"><li>Services</li></Link>
                </ul>
              </div>
            </div>
            <hr className="small" />
          </div>
          <div className="container">
            <div className="row">
              <div className="col-md-8 col-sm-6 col-12">
                <p className="copyright-text">
                  Copyright © 2024 All Rights Reserved by
                  <Link to="/"><span className="logo">PAPER PILOT</span></Link>
                </p>
              </div>
              <div className="col-md-4 col-sm-6 col-12">
                <ul className="social-icons">
                  <li><Link className="facebook" to="https://www.facebook.com/"><Facebook /><i className="fab fa-facebook-f"></i></Link></li>
                  <li><Link className="twitter" to="https://twitter.com/i/flow/login"><Twitter /><i className="fab fa-twitter"></i></Link></li>
                  <li><Link className="dribbble" to="https://dribbble.com/session/new"><Dribbble /><i className="fab fa-dribbble"></i></Link></li>
                  <li><Link className="linkedin" to="https://www.linkedin.com/login"><LinkedIn /><i className="fab fa-linkedin-in"></i></Link></li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </footer>
      <ScrollToTop />
      
    </>
  );
};

export default Footer;
