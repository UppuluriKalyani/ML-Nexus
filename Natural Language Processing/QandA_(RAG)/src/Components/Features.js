import React, { useEffect } from 'react';
import Footer from './Footer';
import './CSS/Features.css';

const Features = () => {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <>
      <section className="features-grid bg-white radius-24 max-w-1100 padding-lg" style={{marginLeft:'200px'}}>
        <h2 className="h2 text-centre">Our most loved ❤️ features</h2>
        <div className="w-layout-grid features-list">
          <div className="feature__highlights">
            <h4 className="h4-features">
              <span className="text-purple-bold">Structured summaries </span>
              from research papers in any format
            </h4>
            <div className="feature-illustration">
              <img src="https://cdn.prod.website-files.com/65388e98a52fc34922751f84/667e37e6d74f8f46d92812c6_summaries-any-format.svg" loading="lazy" alt="" className="image-51"/>
            </div>
          </div>
          <div className="feature__highlights">
            <h4 className="h4-features">
              Save summaries to your library and <span className="text-purple-bold">never lose another article</span>
            </h4>
            <div className="feature-illustration">
              <img src="https://cdn.prod.website-files.com/65388e98a52fc34922751f84/667e37f70b47192f0f07ccc5_folder-icon.svg" loading="lazy" alt="" className="image-42"/>
            </div>
          </div>
          <div className="feature__highlights">
            <h4 className="h4-features">
              Research quality indicators and <span className="text-purple-bold">study comparisons</span>
            </h4>
            <div className="feature-illustration">
              <img src="https://cdn.prod.website-files.com/65388e98a52fc34922751f84/6673b5994b79d44010434b69_study-comparison.svg" loading="lazy" alt="" className="image-52"/>
            </div>
          </div>
          <div className="feature__highlights">
            <h4 className="h4-features">
              Fully <span className="text-purple-bold">formatted bibliographies</span> in just one click
            </h4>
            <div className="feature-illustration">
              <img src="https://cdn.prod.website-files.com/65388e98a52fc34922751f84/6673b5fc63d79366ad155806_citation-illustration.svg" loading="lazy" alt="" className="image-53"/>
            </div>
          </div>
          <div className="feature__highlights">
            <h4 className="h4-features">
              Import and export <span className="text-purple-bold">any number of articles</span>
            </h4>
            <div className="feature-illustration">
              <img src="https://cdn.prod.website-files.com/65388e98a52fc34922751f84/6673b65f44f9c4cbf1e87a91_import-export-illustration.svg" loading="lazy" alt="" className="image-54"/>
            </div>
          </div>
          <div className="feature__highlights">
            <h4 className="h4-features">
              Integrates with your <span className="text-purple-bold">favourite research apps</span>
            </h4>
            <div className="feature-illustration">
              <img src="https://cdn.prod.website-files.com/65388e98a52fc34922751f84/6673b69dad506db779ce07a9_integrations-illustrations.svg" loading="lazy" alt="" className="image-55"/>
            </div>
          </div>
        </div>
      </section>
      <Footer />
    </>
  );
}

export default Features;
