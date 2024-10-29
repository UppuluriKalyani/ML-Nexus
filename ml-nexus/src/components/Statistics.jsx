import React, { useContext, useEffect } from 'react'
import Stats from './Stats'
import { FaStamp, FaStrava } from 'react-icons/fa'
import { repoContext } from '../utils/Context'

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
  return (
    <>
      <div className="p-10 ">
        <h1 className='text-4xl font-bold '>Repository Statistics</h1>
        <div className="flex gap-8 p-10">
          {info && info.map((data, i) => <Stats data={data} key={i} index={i} />)}
        </div>
        <div className="px-20 flex gap-4">
          <div className="basis-1/2 h-[60vh] bg-[#324655] md:w-1/2 px-6 rounded-lg py-4 stats overflow-y-auto">
            <h1 className='text-2xl'>Languages</h1>
            <div className="p-5">
            {langs.length > 0 && langs.map((lang,i) => (<h3 className='text-xl mt-7 ' key={i}>{lang.name}</h3>))}
            </div>
          </div>
          <div className="basis-1/2 h-[60vh] bg-[#324655] md:w-1/2 px-6 rounded-lg py-4 overflow-hidden relative">
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
    </>
  )
}

export default Statistics