import React from "react";
import { Link } from "react-router-dom";
const MyNav = () => {
    return ( 
        <>
        <div class="header-margin1">
        <nav className="navbar navbar-expand-sm navbar-light fixed-top" style={{ backgroundColor: '#7151d9', textAlign: "left", marginTop: "px",boxShadow: '0 8px 16px rgba(0, 0, 0, 0.3)' }}>
          <Link to="/" className="no-underline"><h1 className="navbar-brand brand-name" style={{ fontSize: '3rem',color:'white',marginLeft:'40px' }}>Paper Pilot</h1></Link>
          <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#collapsibleNavId" aria-controls="collapsibleNavId" aria-expanded="false" aria-label="Toggle navigation">
            <span className="navbar-toggler-icon"></span>
          </button>
          
          <div className="collapse navbar-collapse" id="collapsibleNavId">
            <ul className="navbar-nav ms-auto" style={{ backgroundColor: '#7151d9'}}>
            <li className="nav-item illie">
                <Link className="nav-link active" to="/" aria-current="page" style={{ fontSize: '1.75rem',color:'white',marginRight:'16px' }}>Home </Link>
              </li>
              <li className="nav-item illie">
                <Link className="nav-link active" to="/features" style={{ fontSize: '1.75rem',color:'white',marginRight:'16px' }}>Features</Link>
              </li>
              <li className="nav-item illie">
                <Link className="nav-link active" to="/about" style={{ fontSize: '1.75rem',color:'white',marginRight:'16px' }}>About Us</Link>
              </li>
              <li className="nav-item illie">
                <Link className="nav-link active" to="/feedback" style={{ fontSize: '1.75rem',color:'white',marginRight:'16px' }}>Feedback</Link>
              </li>
              
            </ul>
          </div>
        </nav>
      </div>
        </>
     );
}
 
export default MyNav;