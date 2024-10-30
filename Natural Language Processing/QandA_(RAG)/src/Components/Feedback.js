import { useEffect } from "react";
import Footer from "./Footer";
const Feedback = () => {
    useEffect(() => {
        window.scrollTo(0, 0);
    }, []);
    return ( 
        <>
        give us your Feedback


        <Footer/>
        </>
     );
}
 
export default Feedback;