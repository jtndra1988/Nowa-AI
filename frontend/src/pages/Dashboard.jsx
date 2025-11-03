import React from 'react';

// --- STYLES OBJECT ---
// All styles are consolidated here, using the theme from your reference image.
const styles = {
  // Main grid for the dashboard layout
  dashboardGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 320px', // Main content area and a right sidebar
    gridTemplateRows: 'auto 1fr auto', // Header, main content, and bottom table
    gap: '1.5rem',
    height: 'calc(100vh - 3rem)', // Full viewport height minus layout padding
  },
  // Generic card style with the requested colored shadow
  card: {
    backgroundColor: '#2A2F34',
    borderRadius: '12px',
    padding: '1.5rem',
    border: '1px solid rgba(55, 65, 81, 0.5)',
    // The colored shadow effect you requested
    boxShadow: '0 4px 30px rgba(0, 0, 0, 0.1), 0 0 20px rgba(34, 197, 94, 0.1)',
  },
  
  // Header Section
  header: {
    gridColumn: '1 / -1', // Span across both columns
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '1rem',
  },
  headerText: { 
    fontSize: '1.75rem', 
    fontWeight: '600', 
    color: '#E5E7EB' 
  },
  
  // Main Content Area (Left side)
  mainContent: {
    gridColumn: '1 / 2',
    gridRow: '2 / 3',
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '1.5rem',
  },
  statTitle: { 
    color: '#9CA3AF', 
    fontSize: '0.875rem', 
    marginBottom: '0.5rem' 
  },
  statValue: { 
    fontSize: '1.5rem', 
    color: '#E5E7EB', 
    fontWeight: '600' 
  },
  pnlText: (pnl) => ({
    fontSize: '1.5rem',
    color: pnl >= 0 ? '#22C55E' : '#EF4444', // Green for profit, Red for loss
    fontWeight: '600',
  }),
  chartContainer: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#6B7280'
  },
  
  // Right Panel (AI Signal + Order Entry)
  rightPanel: {
    gridColumn: '2 / 3',
    gridRow: '2 / 4', // Span two rows to fill the height
    display: 'flex',
    flexDirection: 'column',
    gap: '2rem', // Increased gap for better separation
  },
  panelTitle: { 
    color: '#E5E7EB', 
    fontWeight: '600', 
    fontSize: '1.1rem', 
    marginBottom: '1rem' 
  },
  aiSignalCard: {
    backgroundColor: 'rgba(34, 197, 94, 0.1)',
    border: '1px solid #22C55E',
    borderRadius: '8px',
    padding: '1rem',
    textAlign: 'center',
  },
  formGroup: { marginBottom: '1rem' },
  label: { 
    display: 'block', 
    color: '#9CA3AF', 
    fontSize: '0.875rem', 
    marginBottom: '0.5rem' 
  },
  input: { 
    width: '100%', 
    boxSizing: 'border-box', 
    backgroundColor: '#1A1E1F', 
    border: '1px solid #374151', 
    borderRadius: '8px', 
    padding: '0.75rem', 
    color: 'white', 
    fontSize: '1rem' 
  },
  buttonGroup: { 
    display: 'grid', 
    gridTemplateColumns: '1fr 1fr', 
    gap: '1rem' 
  },
  button: (variant) => ({
    padding: '0.75rem', 
    fontSize: '1rem', 
    fontWeight: 'bold', 
    borderRadius: '8px', 
    border: 'none', 
    cursor: 'pointer', 
    color: 'white',
    backgroundColor: variant === 'buy' ? '#22C55E' : '#EF4444',
    transition: 'transform 0.2s ease',
  }),
  
  // Bottom Table for Positions
  positionsTableContainer: {
    gridColumn: '1 / 2',
    gridRow: '3 / 4',
  },
  table: { 
    width: '100%', 
    borderCollapse: 'collapse', 
    fontSize: '0.875rem' 
  },
  th: { 
    textAlign: 'left', 
    padding: '0.75rem', 
    color: '#9CA3AF', 
    borderBottom: '1px solid #374151', 
    fontWeight: '500' 
  },
  td: { 
    padding: '0.75rem', 
    borderBottom: '1px solid #374151', 
    verticalAlign: 'middle' 
  },
};

/**
 * Main component for the Trading Terminal Dashboard.
 * This component is composed of smaller, logical sub-components for clarity.
 */
const Dashboard = () => {
  return (
    <div style={styles.dashboardGrid}>
      <DashboardHeader />
      <MainContent />
      <RightPanel />
      <PositionsTable />
    </div>
  );
};

// --- SUB-COMPONENTS ---

const DashboardHeader = () => (
  <header style={styles.header}>
    <h1 style={styles.headerText}>Trading Terminal</h1>
    {/* User Profile component can be added here */}
  </header>
);

const MainContent = () => (
  <main style={styles.mainContent}>
    <PortfolioStats />
    <div style={{...styles.card, ...styles.chartContainer}}>
      <h2>[ TradingView Chart Placeholder: BTC-PERP ]</h2>
    </div>
  </main>
);

const PortfolioStats = () => {
    // Static data representing portfolio metrics
    const totalEquity = 10483.50;
    const unrealizedPNL = 510.00;
    const marginUsed = 10425;

    return (
        <div style={styles.statsGrid}>
            <div style={styles.card}>
                <div style={styles.statTitle}>Total Equity</div>
                <div style={styles.statValue}>${(totalEquity + unrealizedPNL).toLocaleString()}</div>
            </div>
            <div style={styles.card}>
                <div style={styles.statTitle}>Unrealized P&L</div>
                <div style={styles.pnlText(unrealizedPNL)}>+${unrealizedPNL.toLocaleString()}</div>
            </div>
            <div style={styles.card}>
                <div style={styles.statTitle}>Available Margin</div>
                <div style={styles.statValue}>${(totalEquity - marginUsed).toLocaleString()}</div>
            </div>
        </div>
    );
};

const RightPanel = () => (
  <aside style={{...styles.card, ...styles.rightPanel}}>
    <AISignal />
    <OrderEntry />
  </aside>
);

const AISignal = () => (
    <div>
        <h2 style={styles.panelTitle}>AI Trade Signal</h2>
        <div style={styles.aiSignalCard}>
            <p style={{ margin: 0, color: '#9CA3AF', fontSize: '0.8rem' }}>BTC-PERP | 5m</p>
            <p style={{ margin: '0.5rem 0', fontSize: '1.5rem', fontWeight: 'bold', color: '#22C55E' }}>STRONG BUY</p>
            <p style={{ margin: 0, color: '#E5E7EB' }}>Confidence: <span style={{ fontWeight: 'bold' }}>88%</span></p>
        </div>
    </div>
);

const OrderEntry = () => (
  <div>
    <h2 style={styles.panelTitle}>Order Entry</h2>
    <div style={styles.formGroup}>
      <label style={styles.label}>Order Type</label>
      <input style={styles.input} type="text" defaultValue="Market" />
    </div>
    <div style={styles.formGroup}>
      <label style={styles.label}>Size (BTC)</label>
      <input style={styles.input} type="text" defaultValue="0.1" />
    </div>
    <div style={styles.buttonGroup}>
      <button style={styles.button('buy')}>BUY / LONG</button>
      <button style={styles.button('sell')}>SELL / SHORT</button>
    </div>
  </div>
);

const PositionsTable = () => {
    // Static data mimicking the 'Position' model from your backend
    const dummyPositions = [
        { instrument: 'BTC-PERP', side: 'Long', size: 0.5, entry: 68500, mark: 69120, pnl: 310 },
        { instrument: 'ETH-PERP', side: 'Short', size: 10, entry: 3500, mark: 3480, pnl: 200 },
        { instrument: 'SOL-PERP', side: 'Long', size: 50, entry: 145.20, mark: 142.10, pnl: -155 },
    ];

    return (
        <section style={{...styles.card, ...styles.positionsTableContainer}}>
            <h2 style={{...styles.panelTitle, marginTop: 0}}>Open Positions ({dummyPositions.length})</h2>
            <table style={styles.table}>
                <thead>
                    <tr>
                        <th style={styles.th}>Instrument</th>
                        <th style={styles.th}>Side</th>
                        <th style={styles.th}>Size</th>
                        <th style={styles.th}>Entry Price</th>
                        <th style={styles.th}>Mark Price</th>
                        <th style={styles.th}>Unrealized P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {dummyPositions.map((pos, index) => (
                        <tr key={index}>
                            <td style={styles.td}>{pos.instrument}</td>
                            <td style={{ ...styles.td, color: pos.side === 'Long' ? '#22C55E' : '#EF4444' }}>{pos.side}</td>
                            <td style={styles.td}>{pos.size}</td>
                            <td style={styles.td}>${pos.entry.toLocaleString()}</td>
                            <td style={styles.td}>${pos.mark.toLocaleString()}</td>
                            <td style={{ ...styles.td, color: pos.pnl >= 0 ? '#22C55E' : '#EF4444' }}>${pos.pnl.toFixed(2)}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </section>
    );
};

export default Dashboard;

