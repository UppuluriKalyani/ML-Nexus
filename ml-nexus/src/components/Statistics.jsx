import React, { useContext, useEffect } from 'react'
import Stats from './Stats'
import { FaRegWindowMaximize, FaStamp, FaStrava } from 'react-icons/fa'
import { repoContext } from '../utils/Context'
import { SiJupyter } from "react-icons/si";
import { ImHtmlFive2 } from "react-icons/im";
import { TbBrandPython } from "react-icons/tb";
import { TbCircleLetterR } from "react-icons/tb";
import { TbBrandJavascript } from "react-icons/tb";
import { DiCss3 } from "react-icons/di";
import { IoLogoNodejs } from "react-icons/io";
import { VscTerminalPowershell } from "react-icons/vsc";
import { TbBrandTypescript } from "react-icons/tb";
import { TbBrandDocker } from "react-icons/tb";
import { SlPuzzle } from "react-icons/sl";
import Btn from './Btn';



function Statistics() {
  const { projects, info, langs } = useContext(repoContext)
  useEffect(() => {
    console.log(langs)
  })
  const colors = [
    "#A78BFA",
    "#92C9D1",
    "#34D399",
    "#F87171",
    "#FBBF24",
    "#F472B6",
    "#60A5FA",
    "#D4A5A5",


  ]
  const showIcon = (name) => {
    console.log(name)
    switch (name) {
      case "Jupyter Notebook":
       return <SiJupyter size="30px" />
        break;
      case "HTML":
       return <ImHtmlFive2 size="30px" />
      case "Python":
       return <TbBrandPython size="30px" />
        break;
      case "JavaScript":
       return <TbBrandJavascript size="30px" />
        break;
        case "R":
         return <TbCircleLetterR size="30px" />
        break;
        case "CSS":
         return <DiCss3 size="30px" />
        break;
        case "EJS":
         return <IoLogoNodejs size="30px" />
        break;
        case "Shell":
         return <VscTerminalPowershell size="30px" />
        break;
        case "TypeScript":
         return <TbBrandTypescript size="30px" />
        break;
        case "Dockerfile":
         return <TbBrandDocker size="30px" />
        break;
        case "Batchfile":
         return <FaRegWindowMaximize size="30px" />
        break;



      default:
        break;
    }
  }
  return (
    <>
      <div className="p-10 ">
        <h1 className='text-4xl font-bold '>Repository Statistics</h1>
        <div className="flex gap-8 p-10">
          {info && info.map((data, i) => <Stats data={data} key={i} index={i} />)}
        </div>
        <div className="px-20 flex gap-4">
          <div className="basis-1/2 h-[60vh] dark:bg-[#324655] md:w-1/2 px-6 rounded-lg py-4 stats overflow-y-auto">
            <h1 className='text-2xl'>Languages</h1>
            <div className="p-5">
            {langs.length > 0 && langs.map((lang,i) => (
              <>
              <div key={i} className='flex relative items-center mt-7 gap-4'>
              {showIcon(lang.name)}
              
              <h3 className='text-xl' key={i}>{lang.name}</h3>
              </div>
              </>
            ))}
            </div>
          </div>
          <div className="basis-1/2 h-[60vh] dark:bg-[#324655] md:w-1/2 px-6 rounded-lg py-4 overflow-hidden relative">
            <h1 className='text-2xl'>Milestones progress</h1>
            <div className="flex items-start p-4 gap-5 overflow-y-scroll max-h-[50vh] stats stats relative">
             <div className="sticky top-0 left-0">
             <div class="relative bg-black w-64 h-64 rounded-full ">
                <div class="absolute inset-0 rounded-full bg-[conic-gradient(from_0deg,_#A78BFA_0%,_#A78BFA_10%,_#92C9D1_10%,_#92C9D1_20%,_#34D399_20%,_#34D399_25%,_#F87171_25%,_#F87171_40%,_#FBBF24_40%,_#FBBF24_50%,_#f472b6_50%,_#f472b6_70%,_#60A5FA_70%,_#60A5FA_90%,_#D4A5A5_90%,_#D4A5A5_100%)]">
                </div>
              </div>
             </div>
              <div className="flex flex-col text-md gap-6 h-full overflow-y-auto">
                {projects.map((p, i) => (<div className="flex items-center gap-2">
                  <div className={`rounded-full h-6 w-6`}
                  style={{backgroundColor: i > 6 ? colors[7] : colors[i]}}
                  ></div>
                  <h4 className=' text-center'>{p.name}</h4>
                </div>))}

              </div>
            </div>
          </div>

        </div>
       



      </div>
      <div className="flex mt-4">
        <div className="mx-auto h-[30vh] dark:bg-[#3D5966] w-full px-6 flex items-center justify-center">
            <div className="flex flex-col items-center gap-4">
            <h1 className='text-2xl flex items-center gap-3' ><span> <SlPuzzle /> </span> Have Something in Mind ? </h1>
            <Btn value={"Raise issue Now!"}/>
            </div>
          </div>
        </div>
    </>
  )
}

export default Statistics