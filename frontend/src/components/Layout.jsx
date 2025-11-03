import React from 'react';
import Sidebar from './Sidebar';

const styles = {
  layout: {
    display: 'flex',
    minHeight: '100vh',
  },
  mainContent: {
    flex: 1,
    padding: '1.5rem',
  },
};

/**
 * The Layout component provides the main structure for the application,
 * containing the sidebar and the main content area where pages are rendered.
 */
const Layout = ({ children }) => {
  return (
    <div style={styles.layout}>
      <Sidebar />
      <main style={styles.mainContent}>{children}</main>
    </div>
  );
};

export default Layout;
