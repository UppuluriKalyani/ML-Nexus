import React from 'react';
import ResearchPaper from './ResearchPaper';
import BlocksContent1 from './BlocksContent1'
import BlocksContent2 from './BlocksContent2'
import BlocksContent3 from './BlocksContent3'
import Footer from './Footer';
import Random from './Random';

const MainPage = () => {
    return ( 
        <>
            
            <div className="header-margin" style={{marginBottom:'130px'}}>
                <h1 className='text-size1'>Summarize, analyze and</h1>
                <h1 className='text-size1'>do your Research</h1>
            </div>
            <div className="blocks-container">
                <BlocksContent1 />
                <BlocksContent2 />
                <BlocksContent3 />
            </div>
            <ResearchPaper/>
            <Footer/>
        </>
     );
}
 
export default MainPage;
