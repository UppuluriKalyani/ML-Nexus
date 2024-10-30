// import React, { useState } from 'react';
// import FileUpload from './FileUpload';
// import './CSS/Research.css';

// const ResearchPaper = () => {
//     const [summary, setSummary] = useState('');

//     return (
//         <div className="research-paper" style={{ height: '700px', paddingBottom: '1000px' }}>
//             <div className="paper-content">
//                 <div className="browser__frame xl">
//                     <div className="browser__nav large">
//                         <div className="browser__title">Paper Pilot</div>
//                     </div>
//                     <div className="browser__content">
//                         <div className="browser__body">
//                             <img
//                                 src="https://cdn.prod.website-files.com/65388e98a52fc34922751f84/660047cf179511a669c1180c_research__paper.svg"
//                                 loading="lazy"
//                                 alt="Research Paper Illustration"
//                                 className="paper-img"
//                             />
//                             <div className="paper-section">
//                                 <div id="skeleton__abstract" className="skeleton__box">
//                                     <h4 className="h4">Abstract</h4>
//                                     <div className="text__skeleton1"></div>
//                                     <div className="text__skeleton2"></div>
//                                     <div className="text__skeleton display-none"></div>
//                                 </div>
//                                 <div id="skeleton__intro" className="skeleton__box">
//                                     <h4 className="h4">Introduction</h4>
//                                     <div className="text__skeleton1"></div>
//                                     <div className="text__skeleton2"></div>
//                                     <div className="text__skeleton display-none"></div>
//                                 </div>
//                                 <div id="skeleton__methods" className="skeleton__box">
//                                     <h4 className="h4">Methods</h4>
//                                     <div className="text__skeleton1"></div>
//                                     <div className="text__skeleton2"></div>
//                                     <div className="text__skeleton display-none"></div>
//                                 </div>
//                                 <div id="skeleton__results" className="skeleton__box">
//                                     <h4 className="h4">Results</h4>
//                                     <div className="text__skeleton1"></div>
//                                     <div className="text__skeleton2"></div>
//                                     <div className="text__skeleton display-none"></div>
//                                 </div>
//                             </div>
//                             <div className="button-container">
//                                 <FileUpload setSummary={setSummary} />
//                             </div>
//                         </div>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// };

// export default ResearchPaper;

import React from 'react';
import FileUpload from './FileUpload';
import './CSS/Research.css';

const ResearchPaper = () => {
    return (
        <div className="research-paper" style={{ height: '700px', paddingBottom: '1000px' }}>
            <div className="paper-content">
                <div className="browser__frame xl">
                    <div className="browser__nav large">
                        <div className="browser__title">Paper Pilot</div>
                    </div>
                    <div className="browser__content">
                        <div className="browser__body">
                            <img
                                src="https://cdn.prod.website-files.com/65388e98a52fc34922751f84/660047cf179511a669c1180c_research__paper.svg"
                                loading="lazy"
                                alt="Research Paper Illustration"
                                className="paper-img"
                            />
                            <div className="paper-section">
                                <div id="skeleton__abstract" className="skeleton__box">
                                    <h4 className="h4">Abstract</h4>
                                    <div className="text__skeleton1"></div>
                                    <div className="text__skeleton2"></div>
                                    <div className="text__skeleton display-none"></div>
                                </div>
                                <div id="skeleton__intro" className="skeleton__box">
                                    <h4 className="h4">Introduction</h4>
                                    <div className="text__skeleton1"></div>
                                    <div className="text__skeleton2"></div>
                                    <div className="text__skeleton display-none"></div>
                                </div>
                                <div id="skeleton__methods" className="skeleton__box">
                                    <h4 className="h4">Methods</h4>
                                    <div className="text__skeleton1"></div>
                                    <div className="text__skeleton2"></div>
                                    <div className="text__skeleton display-none"></div>
                                </div>
                                <div id="skeleton__results" className="skeleton__box">
                                    <h4 className="h4">Results</h4>
                                    <div className="text__skeleton1"></div>
                                    <div className="text__skeleton2"></div>
                                    <div className="text__skeleton display-none"></div>
                                </div>
                            </div>
                            <div className="button-container">
                                <FileUpload />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ResearchPaper;
