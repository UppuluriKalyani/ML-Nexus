<?php

$state=$_POST["state"];

$xx=exec("sudo python /human_following/master.py $state");

?>