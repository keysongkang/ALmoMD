# Installation

<div class="command-box">
  <pre><code id="command">pip install .</code></pre>
  <button onclick="copyToClipboard()">Copy</button>
</div>

<script>
  function copyToClipboard() {
    var commandText = document.getElementById("command");
    var range = document.createRange();
    range.selectNode(commandText);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    document.execCommand("copy");
    window.getSelection().removeAllRanges();
  }
</script>