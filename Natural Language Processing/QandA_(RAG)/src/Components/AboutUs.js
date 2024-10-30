import React from 'react';
import './CSS/AboutUs.css';
import { useEffect } from 'react';
import Footer from './Footer';
const AboutUs = () => {
  useEffect(() => {
    window.scrollTo(0, 0);
}, []);
  return (
    <>
    <section className="section__box bg-light-purple padding-x-64">
      <div id="header__text" className="w-layout-vflex flex-centre gap-y-32 max-w-960">
        
        <h1 className="h1 purple-05 text-centre">
          Our goal is to make the world’s knowledge more accessible to everyone,{' '}
          <span className="text-purple">by building tools which help you:</span>
        </h1>
        <div className="w-layout-vflex list__custom">
          <div className="w-layout-hflex li__custom">
            <img
              src="https://cdn.prod.website-files.com/65388e98a52fc34922751f84/6582dc31ac055daeab253ef6_enhanced__summary.svg"
              loading="lazy"
              alt="Understand complex topics icon"
              className="size-56"
            />
            <h2 className="h2">Understand complex topics</h2>
          </div>
          <div className="w-layout-hflex li__custom">
            <img
              src="https://cdn.prod.website-files.com/65388e98a52fc34922751f84/65b9c9b88a695f1636e11695_knowledge__icn.svg"
              loading="lazy"
              alt="Keep track of your knowledge icon"
              className="size-56"
            />
            <h2 className="h2">Keep track of your knowledge</h2>
          </div>
          <div className="w-layout-hflex li__custom">
            <img
              src="https://cdn.prod.website-files.com/65388e98a52fc34922751f84/65b9c9b863c0918c60031278_aha__icn.svg"
              loading="lazy"
              alt="Discover more Aha! moments icon"
              className="size-56"
            />
            <h2 className="h2">Discover more Aha! moments</h2>
          </div>
          <div className="w-layout-hflex li__custom">
            <img
              src="https://cdn.prod.website-files.com/65388e98a52fc34922751f84/65b8d83c0a43aef9516990b1_enhance__icn.svg"
              loading="lazy"
              alt="Apply what you’ve learned icon"
              className="size-56"
            />
            <h2 className="h2">Apply what you’ve learned</h2>
          </div>
        </div>
      </div>
    </section>
    <Footer/>
    </>
    
  );
};

export default AboutUs;
