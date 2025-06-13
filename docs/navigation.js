// Insert the navigation sidebar into the page
document.addEventListener('DOMContentLoaded', function() {
  // Check if this is the index page
  const isIndexPage = window.location.pathname.endsWith('index.html') || 
                       window.location.pathname === '/' || 
                       window.location.pathname.endsWith('/');
  
  // Don't add the sidebar to the index.html page
  if (isIndexPage) {
    return;
  }
  
  // Add the tool-page class to the body for non-index pages
  document.body.classList.add('tool-page');
  
  // Determine relative path to root based on current page
  const isInToolsDir = window.location.pathname.includes('/tools/');
  const rootPath = isInToolsDir ? '../' : '';
  
  const sidebar = `
    <div class="sidebar">
      <div class="sidebar-header">
        <a href="${rootPath}index.html" class="logo-link">
          ArcGIS Pro AI Toolbox
        </a>
      </div>
      <div class="sidebar-nav">
        <a href="${rootPath}tools/FeatureLayer.html">Create AI Feature Layer</a>
        <a href="${rootPath}tools/Field.html">Add AI-Generated Field</a>
        <a href="${rootPath}tools/GetMapInfo.html">Get Map Info</a>
        <a href="${rootPath}tools/Python.html">Generate Python Code</a>
        <a href="${rootPath}tools/ConvertTextToNumeric.html">Convert Text to Numeric</a>
        <a href="${rootPath}tools/GenerateTool.html">Generate Tool</a>
      </div>
    </div>
  `;
  
  // Insert the sidebar as the first child of the body
  document.body.insertAdjacentHTML('afterbegin', sidebar);
});