using Microsoft.EntityFrameworkCore;
using WebApplication1.Models; // Make sure this namespace matches where your User model is located

namespace WebApplication1.Data // Change this if your project name is different
{
    public class ApplicationDbContext : DbContext
    {
        public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
            : base(options)
        {
        }

        public DbSet<User> Users { get; set; }
    }
}