<?php

if (isset($_GET['test']) || isset($_GET['option']) || isset($_GET['checkbox'])) {
    $test = $_GET['test'];
    $option = $_GET['option'];
    $checkboxes = $_GET['checkboxes'];

    for ($i = 0; $i < count($test); $i++) {
        $testVal = htmlspecialchars($test[$i]); 
        $optionVal = htmlspecialchars($option[$i]);
        $checkboxVal = isset($checkboxes[$i]) ? 'Checked' : 'Unchecked';

        echo "<h3>Entry " . ($i + 1) . "</h3>";
        echo "Text: $testVal <br>";
        echo "Option: $optionVal <br>";
        echo "Checkbox: $checkboxVal <br>";
    }       
} else {
    echo "No data submitted";
}