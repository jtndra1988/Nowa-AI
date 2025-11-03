import React from 'react';
import { NavLink } from 'react-router-dom';

const styles = {
  sidebar: {
    width: '220px',
    backgroundColor: '#2A2F34', // Dark panel background from theme
    padding: '1.5rem',
    display: 'flex',
    flexDirection: 'column',
    borderRight: '1px solid rgba(55, 65, 81, 0.5)',
    // The colored shadow effect you requested, combining a dark general shadow
    // with a subtle green glow to match the theme's accent color.
    boxShadow: '0 4px 30px rgba(0, 0, 0, 0.1), 0 0 20px rgba(34, 197, 94, 0.1)',
  },
  logo: {
    fontSize: '1.5rem',
    fontWeight: 'bold',
    color: '#E5E7EB',
    marginBottom: '3rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
  },
  logoIcon: {
      width: '32px',
      height: '32px',
      stroke: '#22C55E', // Green accent for the logo icon
  },
  nav: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
  },
  navLink: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    color: '#9CA3AF', // Muted color for inactive links
    textDecoration: 'none',
    fontSize: '0.95rem',
    padding: '0.75rem 1rem',
    borderRadius: '8px',
    transition: 'background-color 0.2s ease, color 0.2s ease',
  },
  activeNavLink: {
    backgroundColor: 'rgba(34, 197, 94, 0.15)', // Subtle green background for active link
    color: '#22C55E', // Vibrant green text for active link
    fontWeight: '600',
  },
  icon: {
    width: '20px',
    height: '20px',
  }
};

/**
 * The Sidebar component provides the main navigation for the application.
 * It uses NavLink to automatically handle active link styling.
 */
const Sidebar = () => {
  // This function conditionally applies the 'activeNavLink' style
  const getNavLinkStyle = ({ isActive }) => {
    return isActive ? { ...styles.navLink, ...styles.activeNavLink } : styles.navLink;
  };

  return (
    <aside style={styles.sidebar}>
      <div style={styles.logo}>
        <svg style={styles.logoIcon} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.896 16.205L16.5 18.75l-.396-2.545a3 3 0 00-2.06-2.06L12 14.25l2.545-.396a3 3 0 002.06-2.06L17.25 10.5l.395 2.545a3 3 0 002.06 2.06L21.75 15l-2.545.395a3 3 0 00-2.06 2.06z" /></svg>
        AI BOT
      </div>
      <nav style={styles.nav}>
        {/* These NavLinks correspond to the pages in our UI blueprint */}
        <NavLink to="/" style={getNavLinkStyle}>
            <svg style={styles.icon} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6a7.5 7.5 0 100 15 7.5 7.5 0 000-15z" /><path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.25-5.25" /></svg>
            Dashboard
        </NavLink>
        <NavLink to="/portfolio" style={getNavLinkStyle}>
            <svg style={styles.icon} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M21 12a2.25 2.25 0 00-2.25-2.25H15a3 3 0 11-6 0H5.25A2.25 2.25 0 003 12m18 0v6a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 18v-6m18 0V9M3 12V9m18 3a2.25 2.25 0 00-2.25-2.25H15a3 3 0 11-6 0H5.25A2.25 2.25 0 003 9m18 3V9" /></svg>
            Portfolio
        </NavLink>
        <NavLink to="/intelligence" style={getNavLinkStyle}>
            <svg style={styles.icon} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 1.5m1-1.5l1 1.5m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6" /></svg>
            Market Intelligence
        </NavLink>
         <NavLink to="/system" style={getNavLinkStyle}>
            <svg style={styles.icon} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25z" /></svg>
            System Health
        </NavLink>
      </nav>
    </aside>
  );
};

export default Sidebar;
