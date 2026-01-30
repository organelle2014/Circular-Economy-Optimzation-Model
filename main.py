import numpy as np
from scipy.optimize import fsolve, minimize_scalar
import matplotlib.pyplot as plt


class ConsumerMarket:    
    def __init__(self, delta=100, k=5, X=0.5, w=10, q=0.8):
        #Initialize the consumer market model.
        self.delta = delta  # Max consumer valuation
        self.k = k  # Mismatch cost
        self.X = X  # Ownership preference intensity
        self.w = w  # Wholesale cost
        self.q = q  # Rental quality factor
        
    def utility_buy(self, v, Ps):
        
        return v - Ps - self.k * (1 - self.X)
    
    def utility_rent(self, v, Pr):
        
        t = 1  # Single rental cycle
        return v * self.q * t - Pr - self.k * self.X
    
    def utility_outside(self):
        
        return 0
    
    def choice_indicator(self, v, Ps, Pr):
        
        uS = self.utility_buy(v, Ps)
        uR = self.utility_rent(v, Pr)
        uOut = self.utility_outside()
        
        utilities = {'buy': uS, 'rent': uR, 'outside': uOut}
        choice = max(utilities, key=utilities.get)
        
        return choice, utilities[choice]
    
    def market_shares(self, Ps, Pr):
    
        # Find threshold types
        # v* where buyer indifferent to outside option: v* - Ps - k(1-X) = 0
        vBuyThreshold = Ps + self.k * (1 - self.X)
        
        # v** where renter indifferent to outside option: v**q - Pr - kX = 0
        vRentThreshold = (Pr + self.k * self.X) / (self.q)
        
        # Clamp thresholds to valid range
        vBuyThreshold = np.clip(vBuyThreshold, 0, self.delta)
        vRentThreshold = np.clip(vRentThreshold, 0, self.delta)
        
        
        if self.q < 1:
            vIndiff = (Ps - Pr - self.k + 2*self.k*self.X) / (1 - self.q)
            vIndiff = np.clip(vIndiff, 0, self.delta)
        else:
            vIndiff = self.delta  # If q=1, renting dominates for all consumers
        
        # Calculate shares based on consumer thresholds
        dBuy = 0
        dRent = 0
        dOutside = 0
        
        # Case 1: No one buys or rents if thresholds are out of range
        if vBuyThreshold >= self.delta and vRentThreshold >= self.delta:
            dOutside = 1.0
        else:
            # Determine market segments
            if vIndiff <= vBuyThreshold:
                # Everyone who participates rents
                if vRentThreshold <= self.delta:
                    dRent = (self.delta - vRentThreshold) / self.delta
                dOutside = (vRentThreshold) / self.delta
            elif vIndiff >= vRentThreshold:
                # Everyone who participates buys
                if vBuyThreshold <= self.delta:
                    dBuy = (self.delta - vBuyThreshold) / self.delta
                dOutside = (vBuyThreshold) / self.delta
            else:
                # Mixed market: both buying and renting occur
                if vBuyThreshold <= vIndiff:
                    dBuy = (self.delta - vIndiff) / self.delta
                if vRentThreshold <= vIndiff:
                    dRent = (vIndiff - vRentThreshold) / self.delta
                dOutside = max(0, vRentThreshold / self.delta)
        
        # Ensure shares sum to 1
        total = dBuy + dRent + dOutside
        if total > 0:
            dBuy /= total
            dRent /= total
            dOutside /= total
        
        return {
            'buy': dBuy,
            'rent': dRent,
            'outside': dOutside,
            'vBuyThreshold': vBuyThreshold,
            'vRentThreshold': vRentThreshold,
            'vIndiff': vIndiff if self.q < 1 else None
        }
    
    def seller_profit(self, Ps, Pr):
        """
        Seller's profit from selling.
        
        π_s = (Ps - w) * Ds(Ps, Pr)
        
        Parameters:
        -----------
        Ps : float
            Selling price
        Pr : float
            Rental price
            
        Returns:
        --------
        float : Seller's profit
        """
        shares = self.market_shares(Ps, Pr)
        return (Ps - self.w) * shares['buy']
    
    def renter_profit(self, Pr, Ps):
        """
        Renter's profit from renting.
        
        π_r = (Pr - w) * D_r(Ps, Pr)
        Note: For simplicity, assuming same wholesale cost w (no scaling for durability)
        
        Parameters:
        -----------
        Pr : float
            Rental price
        Ps : float
            Selling price
            
        Returns:
        --------
        float : Renter's profit
        """
        shares = self.market_shares(Ps, Pr)
        return (Pr - self.w) * shares['rent']
    
    def consumer_surplus(self, Ps, Pr):
        """
        Calculate total consumer surplus.
        
        Returns:
        --------
        float : Total consumer surplus from all consumers
        """
        shares = self.market_shares(Ps, Pr)
        
        # Integrate utility for each segment
        cs = 0
        
        # Buying segment
        vBuyStart = Ps + self.k * (1 - self.X)
        if vBuyStart < self.delta and shares['buy'] > 0:
            # CS = ∫[vBuyStart to delta] (v - Ps - k(1-X)) dv
            cs += (self.delta**2 - vBuyStart**2) / 2 - Ps * (self.delta - vBuyStart) - self.k * (1 - self.X) * (self.delta - vBuyStart)
        
        # Renting segment
        vRentStart = (Pr + self.k * self.X) / self.q
        if shares['rent'] > 0 and vRentStart < self.delta:
            vUpper = self.delta
            if shares['buy'] > 0 and shares['vIndiff'] is not None:
                vUpper = shares['vIndiff']
            if vUpper > vRentStart:
                # CS = ∫[vRentStart to vUpper] (v*q - Pr - kX) dv
                cs += (
                    self.q * (vUpper**2 - vRentStart**2) / 2
                    - (Pr + self.k * self.X) * (vUpper - vRentStart)
                )
        
        return cs
    
    def total_welfare(self, Ps, Pr):
        """
        Calculate total welfare (producer surplus + consumer surplus).
        
        Returns:
        --------
        float : Total welfare
        """
        psSeller = self.seller_profit(Ps, Pr)
        psRenter = self.renter_profit(Pr, Ps)
        cs = self.consumer_surplus(Ps, Pr)
        
        return psSeller + psRenter + cs
    
    def total_profit(self, Ps, Pr):
        """
        Calculate total profit (seller + renter).
        
        Returns:
        --------
        float : Total profit
        """
        return self.seller_profit(Ps, Pr) + self.renter_profit(Pr, Ps)


class MarketEquilibrium:
    """
    Find and analyze market equilibrium under different competition structures.
    """
    
    def __init__(self, market: ConsumerMarket):
        """
        Initialize equilibrium solver.
        
        Parameters:
        -----------
        market : ConsumerMarket
            The consumer market model
        """
        self.market = market
    
    def monopoly_equilibrium(self, mode='sequential'):
        """
        Find equilibrium under monopoly (single firm selling one product type).
        
        Parameters:
        -----------
        mode : str
            'sequential': Stackelberg game (one firm moves first)
            'simultaneous': Bertrand competition
            
        Returns:
        --------
        dict : Equilibrium prices and market outcomes
        """
        if mode == 'sequential':
            # Seller moves first, renter moves second (Stackelberg)
            return self._stackelberg_equilibrium()
        else:
            return self._bertrand_equilibrium()
    
    def _stackelberg_equilibrium(self):
        """
        Solve Stackelberg game where seller moves first.
        """
        bestSellerProfit = -np.inf
        bestPrices = None
        
        # Grid search over seller's choice of Ps
        for Ps in np.linspace(self.market.w + 1, self.market.delta + self.market.w, 100):
            # Renter's best response to Ps
            result = minimize_scalar(
                lambda Pr: -self.market.renter_profit(Pr, Ps),
                bounds=(self.market.w, Ps),
                method='bounded'
            )
            prBr = result.x
            
            # Seller's profit at this outcome
            profitS = self.market.seller_profit(Ps, prBr)
            
            if profitS > bestSellerProfit:
                bestSellerProfit = profitS
                bestPrices = (Ps, prBr)
        
        Ps, Pr = bestPrices
        shares = self.market.market_shares(Ps, Pr)
        
        return {
            'Ps': Ps,
            'Pr': Pr,
            'Ds': shares['buy'],
            'Dr': shares['rent'],
            'dOut': shares['outside'],
            'profitSeller': self.market.seller_profit(Ps, Pr),
            'profitRenter': self.market.renter_profit(Pr, Ps),
            'totalWelfare': self.market.total_welfare(Ps, Pr)
        }
    
    def _bertrand_equilibrium(self):
        """
        Solve Bertrand (simultaneous move) game.
        """
        def equilibrium_conditions(prices):
            Ps, Pr = prices
            # Enforce pricing constraint Ps > Pr
            if Ps <= Pr:
                return [1e6, 1e6]
            
            # Best response conditions (derivatives of profit)
            # For simplicity, use finite differences
            eps = 0.01
            
            profitSUp = self.market.seller_profit(Ps + eps, Pr)
            profitSDown = self.market.seller_profit(Ps - eps, Pr)
            drDPs = (profitSUp - profitSDown) / (2 * eps)
            
            profitRUp = self.market.renter_profit(Pr + eps, Ps)
            profitRDown = self.market.renter_profit(Pr - eps, Ps)
            drDPr = (profitRUp - profitRDown) / (2 * eps)
            
            return [drDPs, drDPr]
        
        # Initial guess
        psInit = self.market.w + (self.market.delta - self.market.w) * 0.6
        prInit = self.market.w + (self.market.delta - self.market.w) * 0.3
        
        try:
            solution = fsolve(equilibrium_conditions, [psInit, prInit])
            Ps, Pr = solution
        except:
            Ps, Pr = psInit, prInit
        
        # Ensure Ps > Pr
        if Ps <= Pr:
            Ps, Pr = max(Ps, Pr) + 1, min(Ps, Pr)
        
        shares = self.market.market_shares(Ps, Pr)
        
        return {
            'Ps': Ps,
            'Pr': Pr,
            'Ds': shares['buy'],
            'Dr': shares['rent'],
            'dOut': shares['outside'],
            'profitSeller': self.market.seller_profit(Ps, Pr),
            'profitRenter': self.market.renter_profit(Pr, Ps),
            'totalWelfare': self.market.total_welfare(Ps, Pr)
        }
    
    def integrated_channel(self):
        """
        Integrated channel: Single firm offers both buying and renting.
        Maximizes total profit = profitSeller + profitRenter
        """
        bestTotalProfit = -np.inf
        bestPrices = None
        
        # Grid search over both prices
        for Ps in np.linspace(self.market.w + 1, self.market.delta, 50):
            for Pr in np.linspace(self.market.w + 0.1, Ps - 0.1, 50):
                totalProfit = self.market.total_profit(Ps, Pr)
                if totalProfit > bestTotalProfit:
                    bestTotalProfit = totalProfit
                    bestPrices = (Ps, Pr)
        
        if bestPrices is None:
            bestPrices = (self.market.w + 20, self.market.w + 10)
        
        Ps, Pr = bestPrices
        shares = self.market.market_shares(Ps, Pr)
        
        return {
            'Ps': Ps,
            'Pr': Pr,
            'Ds': shares['buy'],
            'Dr': shares['rent'],
            'dOut': shares['outside'],
            'profitTotal': self.market.total_profit(Ps, Pr),
            'totalWelfare': self.market.total_welfare(Ps, Pr)
        }


def demonstrate_market():
    """
    Demonstrate the consumer market model with example parameters.
    """
    print("=" * 70)
    print("HETEROGENEOUS CONSUMER MARKET MODEL")
    print("=" * 70)
    
    # Create market
    market = ConsumerMarket(
        delta=100,      # Max consumer type
        k=5,            # Mismatch cost
        X=0.5,          # Ownership preference intensity
        w=10,           # Wholesale cost
        q=0.8           # Rental quality factor
    )
    
    print(f"\nMarket Parameters:")
    print(f"  δ (Max valuation): {market.delta}")
    print(f"  k (Mismatch cost): {market.k}")
    print(f"  X (Ownership intensity): {market.X}")
    print(f"  w (Wholesale cost): {market.w}")
    print(f"  q (Rental quality): {market.q}")
    
    # Example: given prices
    print(f"\n{'-' * 70}")
    print("EXAMPLE: Market Outcomes at Specific Prices")
    print(f"{'-' * 70}")
    
    Ps, Pr = 35, 20
    shares = market.market_shares(Ps, Pr)
    
    print(f"\nPrices: Ps = {Ps}, Pr = {Pr}")
    print(f"Market Shares:")
    print(f"  dBuy (Buying):  {shares['buy']:.1%}")
    print(f"  dRent (Renting): {shares['rent']:.1%}")
    print(f"  dOutside:       {shares['outside']:.1%}")
    print(f"\nThresholds:")
    print(f"  v* (buy threshold):  {shares['vBuyThreshold']:.2f}")
    print(f"  v** (rent threshold): {shares['vRentThreshold']:.2f}")
    if shares['vIndiff'] is not None:
        print(f"  v*** (indifference):  {shares['vIndiff']:.2f}")
    
    print(f"\nProfits:")
    print(f"  Seller: {market.seller_profit(Ps, Pr):.2f}")
    print(f"  Renter: {market.renter_profit(Pr, Ps):.2f}")
    
    # Find equilibrium
    print(f"\n{'-' * 70}")
    print("CHANNEL STRUCTURE COMPARISON")
    print(f"{'-' * 70}")
    
    eq_solver = MarketEquilibrium(market)
    
    # Independent channel: Stackelberg (seller leads)
    eqIndependent = eq_solver.monopoly_equilibrium(mode='sequential')
    
    print(f"\nA) INDEPENDENT CHANNEL (Separate Seller & Renter):")
    print(f"   Stackelberg Equilibrium (Seller moves first)")
    print(f"   Ps = {eqIndependent['Ps']:.2f}, Pr = {eqIndependent['Pr']:.2f}")
    print(f"   Market Shares: Ds = {eqIndependent['Ds']:.1%}, Dr = {eqIndependent['Dr']:.1%}, dOut = {eqIndependent['dOut']:.1%}")
    print(f"   Seller Profit: {eqIndependent['profitSeller']:.2f}")
    print(f"   Renter Profit: {eqIndependent['profitRenter']:.2f}")
    print(f"   Total Profit:  {eqIndependent['profitSeller'] + eqIndependent['profitRenter']:.2f}")
    print(f"   Total Welfare: {eqIndependent['totalWelfare']:.2f}")
    
    # Integrated channel
    eqIntegrated = eq_solver.integrated_channel()
    
    print(f"\nB) INTEGRATED CHANNEL (Single Firm Offers Both):")
    print(f"   Ps = {eqIntegrated['Ps']:.2f}, Pr = {eqIntegrated['Pr']:.2f}")
    print(f"   Market Shares: Ds = {eqIntegrated['Ds']:.1%}, Dr = {eqIntegrated['Dr']:.1%}, dOut = {eqIntegrated['dOut']:.1%}")
    print(f"   Total Profit:  {eqIntegrated['profitTotal']:.2f}")
    print(f"   Total Welfare: {eqIntegrated['totalWelfare']:.2f}")
    
    # Comparison
    print(f"\n{'-' * 70}")
    print("COMPARISON:")
    print(f"{'-' * 70}")
    profitGain = eqIntegrated['profitTotal'] - (eqIndependent['profitSeller'] + eqIndependent['profitRenter'])
    welfareGain = eqIntegrated['totalWelfare'] - eqIndependent['totalWelfare']
    
    baseProfit = eqIndependent['profitSeller'] + eqIndependent['profitRenter']
    baseWelfare = eqIndependent['totalWelfare']
    
    def format_change(delta, base):
        if abs(base) < 1e-6:
            return f"{delta:.2f} (n/a)"
        return f"{delta:.2f} ({delta / base * 100:.1f}%)"
    
    print(f"   Profit Increase (Integration):  {format_change(profitGain, baseProfit)}")
    print(f"   Welfare Change (Integration):   {format_change(welfareGain, baseWelfare)}")
    print(f"   Price Change: ΔPs = {eqIntegrated['Ps'] - eqIndependent['Ps']:.2f}, ΔPr = {eqIntegrated['Pr'] - eqIndependent['Pr']:.2f}")
    
    return market, eqIndependent, eqIntegrated


if __name__ == '__main__':
    market, eqIndependent, eqIntegrated = demonstrate_market()
